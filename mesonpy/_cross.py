# SPDX-FileCopyrightText: 2022 The meson-python developers
#
# SPDX-License-Identifier: MIT

"""Support for cross builds driven by a PEP 739 ``build-details.json``."""

from __future__ import annotations

import json
import os
import pathlib
import re
import shlex
import textwrap
import typing

from typing import Any, Dict, List, Mapping, Optional, Sequence


if typing.TYPE_CHECKING:  # pragma: no cover
    from mesonpy._compat import Path


# Map from build-details.json ``platform`` prefix to Meson ``host_machine.system``.
_MESON_SYSTEMS = {
    'linux': 'linux',
    'macosx': 'darwin',
    'ios': 'ios',
    'android': 'android',
    'win': 'windows',
    'win32': 'windows',
    'emscripten': 'emscripten',
}

# Map from CPU name to Meson ``host_machine.cpu_family``.
# https://mesonbuild.com/Reference-tables.html#cpu-families-table
_MESON_CPU_FAMILIES = {
    'x86_64': 'x86_64',
    'amd64': 'x86_64',
    'AMD64': 'x86_64',
    'i386': 'x86',
    'i686': 'x86',
    'x86': 'x86',
    'aarch64': 'aarch64',
    'arm64': 'aarch64',
    'ARM64': 'aarch64',
    'arm64_v8a': 'aarch64',
    'armv7l': 'arm',
    'armeabi_v7a': 'arm',
    'armv8l': 'arm',
    'ppc64le': 'ppc64',
    'ppc64': 'ppc64',
    'ppc': 'ppc',
    's390x': 's390x',
    'riscv64': 'riscv64',
    'loongarch64': 'loongarch64',
    'mips64': 'mips64',
    'mips': 'mips',
    'wasm32': 'wasm32',
}

_BIG_ENDIAN_CPUS = {'s390x', 'ppc64', 'ppc', 'mips', 'mips64'}

# Map from toolchain variable names (as in the environment) to Meson
# ``[binaries]`` entries.
_MESON_BINARIES = {
    'CC': 'c',
    'CXX': 'cpp',
    'AR': 'ar',
    'AS': 'as',
    'LD': 'ld',
    'NM': 'nm',
    'RANLIB': 'ranlib',
    'STRIP': 'strip',
    'OBJCOPY': 'objcopy',
    'OBJDUMP': 'objdump',
    'READELF': 'readelf',
    'PKG_CONFIG': 'pkg-config',
    'FC': 'fortran',
    'OBJC': 'objc',
    'OBJCXX': 'objcpp',
    'RUSTC': 'rust',
    'WINDRES': 'windres',
}

# Map from toolchain flag variables to Meson ``[built-in options]``.
_MESON_FLAGS = {
    'CFLAGS': ['c_args'],
    'CXXFLAGS': ['cpp_args'],
    'LDFLAGS': ['c_link_args', 'cpp_link_args'],
}


def _multiarch(build_details: Mapping[str, Any]) -> Optional[str]:
    multiarch = build_details.get('implementation', {}).get('_multiarch')
    if isinstance(multiarch, str) and multiarch:
        return multiarch
    ext_suffix = build_details.get('abi', {}).get('extension_suffix', '')
    # .cpython-314-aarch64-linux-gnu.so -> aarch64-linux-gnu
    match = re.match(r'\.cpython-\d+[a-z]*-(.+)\.(so|pyd|dylib)$', ext_suffix)
    if match and '-' in match.group(1):
        return match.group(1)
    return None


class CrossBuild:
    """Information about the host Python extracted from ``build-details.json``.

    The build frontend passes this information to the ``cross_build_wheel``
    hook; this class interprets it for Meson and for wheel tag computation.
    """

    def __init__(self,
                 build_details: Mapping[str, Any],
                 host_requires_directory: Optional[str] = None,
                 toolchain: Optional[Mapping[str, str]] = None) -> None:
        self.build_details = dict(build_details)
        self.host_requires_directory = pathlib.Path(host_requires_directory) if host_requires_directory else None
        self.toolchain: Dict[str, str] = dict(toolchain or {})

        for key in ('language', 'implementation', 'abi'):
            if key not in self.build_details:
                raise ValueError(f'build-details.json is missing the {key!r} key')

        self.multiarch = _multiarch(self.build_details)
        self.sysconfig_platform: str = self.build_details.get('platform') or ''
        if not self.sysconfig_platform:
            # python-build-standalone ships an empty platform key; reconstruct
            # it from the multiarch tuple.
            if self.multiarch is None:
                raise ValueError('build-details.json has an empty "platform" and no "implementation._multiarch"')
            cpu, os_, *_ = self.multiarch.split('-') + ['']
            if os_.startswith('linux'):
                self.sysconfig_platform = f'linux-{cpu}'
            elif os_.startswith('emscripten'):
                self.sysconfig_platform = f'emscripten-{cpu}'
            else:
                raise ValueError(f'cannot determine host platform from multiarch {self.multiarch!r}')

    # -- Python properties --------------------------------------------------

    @property
    def implementation_name(self) -> str:
        return str(self.build_details['implementation']['name'])

    @property
    def is_pypy(self) -> bool:
        return self.implementation_name == 'pypy'

    @property
    def version_info(self) -> tuple[int, int]:
        info = self.build_details['language'].get('version_info')
        if info:
            return (int(info['major']), int(info['minor']))
        major, minor = str(self.build_details['language']['version']).split('.')[:2]
        return (int(major), int(minor))

    @property
    def python_version(self) -> str:
        """Full version string, e.g. ``3.14.8``."""
        info = self.build_details['language'].get('version_info')
        if info:
            version = f'{info["major"]}.{info["minor"]}.{info.get("micro", 0)}'
            level = info.get('releaselevel', 'final')
            if level != 'final':
                version += f'{level[0]}{info.get("serial", 0)}'
            return version
        return str(self.build_details['language']['version'])

    @property
    def abiflags(self) -> str:
        return ''.join(self.build_details['abi'].get('flags', []))

    @property
    def is_free_threaded(self) -> bool:
        return 't' in self.abiflags

    @property
    def extension_suffix(self) -> str:
        return str(self.build_details['abi']['extension_suffix'])

    # -- Platform properties ------------------------------------------------

    @property
    def system(self) -> str:
        """Meson ``host_machine.system()`` value."""
        prefix = self.sysconfig_platform.split('-', 1)[0]
        return _MESON_SYSTEMS.get(prefix, prefix)

    @property
    def cpu(self) -> str:
        """Meson ``host_machine.cpu()`` value."""
        platform = self.sysconfig_platform
        if platform == 'win32':
            return 'x86'
        if platform.startswith('win-'):
            return platform.split('-', 1)[1].replace('amd64', 'x86_64').replace('arm64', 'aarch64')
        if platform.startswith('linux-'):
            return platform.split('-', 1)[1]
        if platform.startswith('android-'):
            # android-24-arm64_v8a
            return {'arm64_v8a': 'aarch64', 'armeabi_v7a': 'armv7l', 'x86': 'i686'}.get(
                platform.rsplit('-', 1)[1], platform.rsplit('-', 1)[1])
        if platform.startswith('ios-'):
            # ios-13.0-arm64-iphoneos
            return platform.split('-')[2]
        # macosx-11.0-arm64, emscripten-4.0.9-wasm32
        return platform.rsplit('-', 1)[1]

    @property
    def cpu_family(self) -> str:
        """Meson ``host_machine.cpu_family()`` value."""
        cpu = self.cpu
        if cpu in _MESON_CPU_FAMILIES:
            return _MESON_CPU_FAMILIES[cpu]
        if cpu.startswith('arm'):
            return 'arm'
        return cpu

    @property
    def endian(self) -> str:
        return 'big' if self.cpu_family in _BIG_ENDIAN_CPUS else 'little'

    @property
    def binary_format(self) -> str:
        """``'macho'``, ``'pe'`` or ``'elf'``."""
        if self.system in ('darwin', 'ios'):
            return 'macho'
        if self.system in ('windows', 'cygwin'):
            return 'pe'
        return 'elf'

    # -- Wheel tags ---------------------------------------------------------

    @property
    def interpreter_tag(self) -> str:
        from mesonpy._tags import INTERPRETERS
        name = INTERPRETERS.get(self.implementation_name, self.implementation_name)
        major, minor = self.version_info
        return f'{name}{major}{minor}'

    @property
    def abi_tag(self) -> str:
        from mesonpy._tags import abi_tag_from_extension_suffix
        abi = abi_tag_from_extension_suffix(self.extension_suffix)
        if abi is None:
            if self.implementation_name != 'cpython':
                raise NotImplementedError(f'cannot determine ABI tag for {self.implementation_name}')
            major, minor = self.version_info
            abi = f'cp{major}{minor}{self.abiflags}'
        return abi

    @property
    def platform_tag(self) -> str:
        platform = self.sysconfig_platform
        if platform.startswith('macosx-'):
            # macosx-11.0-arm64: normalise the version the way _get_macosx_platform_tag() does.
            _, version, arch = platform.split('-', 2)
            major, minor = (tuple(map(int, version.split('.'))) + (0, 0))[:2]
            if arch == 'arm64' and (major, minor) < (11, 0):
                major, minor = 11, 0
            if major >= 11:
                minor = 0
            return f'macosx_{major}_{minor}_{arch}'
        return platform.replace('-', '_').replace('.', '_').lower()

    # -- Meson integration --------------------------------------------------

    def _resolve_path(self, path: str) -> str:
        """Resolve a path from build-details.json, which may be relative to ``base_prefix``."""
        if os.path.isabs(path):
            return path
        base_prefix = self.build_details.get('base_prefix', '')
        return os.path.normpath(os.path.join(base_prefix, path))

    @property
    def python_pkgconfig_path(self) -> Optional[str]:
        """Directory containing the host Python's ``python3.pc``."""
        path = self.build_details.get('c_api', {}).get('pkgconfig_path')
        return self._resolve_path(path) if path else None

    @property
    def python_include_dir(self) -> Optional[str]:
        """Directory containing the host Python's ``Python.h``."""
        path = self.build_details.get('c_api', {}).get('headers')
        return self._resolve_path(path) if path else None

    def pkg_config_paths(self) -> List[str]:
        """pkg-config search path for the host: host requirements and the host Python."""
        paths = []
        if self.host_requires_directory is not None and self.host_requires_directory.is_dir():
            found = set()
            for pc_file in self.host_requires_directory.rglob('*.pc'):
                if pc_file.parent.name == 'pkgconfig':
                    found.add(os.fspath(pc_file.parent))
            paths.extend(sorted(found))
        if self.python_pkgconfig_path and os.path.isdir(self.python_pkgconfig_path):
            paths.append(self.python_pkgconfig_path)
        return paths

    def write_build_details(self, path: Path) -> None:
        """Write ``build-details.json`` for Meson's ``python.build_config`` option.

        PEP 739 allows ``base_prefix`` to be relative to the location of the
        file; since we write a copy elsewhere, it must already be absolute.
        """
        base_prefix = self.build_details.get('base_prefix')
        if isinstance(base_prefix, str) and not os.path.isabs(base_prefix):
            raise ValueError('build-details.json "base_prefix" must be an absolute path for cross builds')
        pathlib.Path(path).write_text(json.dumps(self.build_details, indent=2), encoding='utf-8')

    def cross_file(self, build_details_path: Path) -> str:
        """Generate the content of a Meson cross file."""

        def quote(value: str) -> str:
            return repr(value)

        def quote_list(values: Sequence[str]) -> str:
            return '[' + ', '.join(repr(v) for v in values) + ']'

        binaries = []
        for var, name in _MESON_BINARIES.items():
            value = self.toolchain.get(var)
            if value:
                argv = shlex.split(value)
                binaries.append(f'{name} = {quote(argv[0]) if len(argv) == 1 else quote_list(argv)}')
        if 'PKG_CONFIG' not in self.toolchain:
            # dependency() lookups for the host machine require pkg-config to
            # be listed in the cross file.
            binaries.append("pkg-config = 'pkg-config'")

        options = [f'python.build_config = {quote(os.fspath(build_details_path))}']
        pkg_config_paths = self.pkg_config_paths()
        if pkg_config_paths:
            options.append(f'pkg_config_path = {quote_list(pkg_config_paths)}')
        for var, names in _MESON_FLAGS.items():
            value = self.toolchain.get(var)
            if value:
                for name in names:
                    options.append(f'{name} = {quote_list(shlex.split(value))}')

        properties = ['needs_exe_wrapper = true']
        if self.host_requires_directory is not None:
            properties.append(f'host_requires_directory = {quote(os.fspath(self.host_requires_directory))}')
        # Do not let pkg-config look into the build machine's system
        # directories when resolving dependencies for the host machine: only
        # search our directories and whatever the toolchain declares.
        pkg_config_libdir = list(pkg_config_paths)
        for var in ('PKG_CONFIG_LIBDIR', 'PKG_CONFIG_PATH'):
            value = self.toolchain.get(var) or os.environ.get(var)
            if value:
                pkg_config_libdir.extend(p for p in value.split(os.pathsep) if p)
        if pkg_config_libdir:
            properties.append(f'pkg_config_libdir = {quote_list(pkg_config_libdir)}')

        return textwrap.dedent('''
            [binaries]
            {binaries}

            [host_machine]
            system = {system}
            cpu_family = {cpu_family}
            cpu = {cpu}
            endian = {endian}

            [properties]
            {properties}

            [built-in options]
            {options}
        ''').format(
            binaries='\n'.join(binaries),
            system=quote(self.system),
            cpu_family=quote(self.cpu_family),
            cpu=quote(self.cpu),
            endian=quote(self.endian),
            properties='\n'.join(properties),
            options='\n'.join(options),
        ).lstrip()
