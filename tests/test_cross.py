# SPDX-FileCopyrightText: 2022 The meson-python developers
#
# SPDX-License-Identifier: MIT

import json
import os
import shutil
import sys
import sysconfig

import pytest

import mesonpy
import mesonpy._cross

from .conftest import MESON_VERSION, chdir, in_git_repo_context
from .test_tags import wheel_builder_test_factory


def build_details(platform='linux-aarch64', *, version=(3, 14, 8), flags=(), multiarch='aarch64-linux-gnu',
                  extension_suffix='.cpython-314-aarch64-linux-gnu.so', implementation='cpython',
                  base_prefix='/opt/host-python'):
    major, minor, micro = version
    details = {
        'schema_version': '1.0',
        'base_prefix': base_prefix,
        'base_interpreter': f'./bin/python{major}.{minor}',
        'platform': platform,
        'language': {
            'version': f'{major}.{minor}',
            'version_info': {'major': major, 'minor': minor, 'micro': micro, 'releaselevel': 'final', 'serial': 0},
        },
        'implementation': {
            'name': implementation,
            'cache_tag': f'{implementation}-{major}{minor}',
            'version': {'major': major, 'minor': minor, 'micro': micro, 'releaselevel': 'final', 'serial': 0},
        },
        'abi': {
            'flags': list(flags),
            'extension_suffix': extension_suffix,
            'stable_abi_suffix': '.abi3.so',
        },
        'libpython': {
            'dynamic': f'./lib/libpython{major}.{minor}.so',
            'link_extensions': False,
        },
        'c_api': {
            'headers': f'./include/python{major}.{minor}',
            'pkgconfig_path': './lib/pkgconfig',
        },
    }
    if multiarch is not None:
        details['implementation']['_multiarch'] = multiarch
    return details


@pytest.mark.parametrize(('details', 'system', 'cpu', 'cpu_family', 'binary_format', 'tag'), [
    (build_details(), 'linux', 'aarch64', 'aarch64', 'elf', 'cp314-cp314-linux_aarch64'),
    # python-build-standalone ships an empty platform key
    (build_details(''), 'linux', 'aarch64', 'aarch64', 'elf', 'cp314-cp314-linux_aarch64'),
    (build_details('linux-x86_64', multiarch='x86_64-linux-gnu', extension_suffix='.cpython-314-x86_64-linux-gnu.so'),
     'linux', 'x86_64', 'x86_64', 'elf', 'cp314-cp314-linux_x86_64'),
    (build_details('linux-armv7l', multiarch='arm-linux-gnueabihf', extension_suffix='.cpython-314-arm-linux-gnueabihf.so'),
     'linux', 'armv7l', 'arm', 'elf', 'cp314-cp314-linux_armv7l'),
    (build_details('linux-aarch64', flags=['t'], extension_suffix='.cpython-314t-aarch64-linux-gnu.so'),
     'linux', 'aarch64', 'aarch64', 'elf', 'cp314-cp314t-linux_aarch64'),
    (build_details('macosx-11.0-arm64', multiarch='darwin', extension_suffix='.cpython-314-darwin.so'),
     'darwin', 'arm64', 'aarch64', 'macho', 'cp314-cp314-macosx_11_0_arm64'),
    (build_details('macosx-10.9-x86_64', multiarch='darwin', extension_suffix='.cpython-314-darwin.so'),
     'darwin', 'x86_64', 'x86_64', 'macho', 'cp314-cp314-macosx_10_9_x86_64'),
    (build_details('ios-13.0-arm64-iphoneos', multiarch='arm64-iphoneos', extension_suffix='.cpython-314-arm64-iphoneos.so'),
     'ios', 'arm64', 'aarch64', 'macho', 'cp314-cp314-ios_13_0_arm64_iphoneos'),
    (build_details('android-24-arm64_v8a', multiarch='aarch64-linux-android',
                   extension_suffix='.cpython-314-aarch64-linux-android.so'),
     'android', 'aarch64', 'aarch64', 'elf', 'cp314-cp314-android_24_arm64_v8a'),
    (build_details('win-amd64', multiarch=None, extension_suffix='.cp314-win_amd64.pyd'),
     'windows', 'x86_64', 'x86_64', 'pe', 'cp314-cp314-win_amd64'),
    (build_details('win-arm64', multiarch=None, extension_suffix='.cp314-win_arm64.pyd'),
     'windows', 'aarch64', 'aarch64', 'pe', 'cp314-cp314-win_arm64'),
    (build_details('emscripten-4.0.9-wasm32', multiarch='wasm32-emscripten',
                   extension_suffix='.cpython-314-wasm32-emscripten.so'),
     'emscripten', 'wasm32', 'wasm32', 'elf', 'cp314-cp314-emscripten_4_0_9_wasm32'),
    (build_details('linux-x86_64', version=(3, 11, 13), implementation='pypy', multiarch='x86_64-linux-gnu',
                   extension_suffix='.pypy311-pp73-x86_64-linux-gnu.so'),
     'linux', 'x86_64', 'x86_64', 'elf', 'pp311-pypy311_pp73-linux_x86_64'),
])
def test_cross_build_properties(details, system, cpu, cpu_family, binary_format, tag):
    cross = mesonpy._cross.CrossBuild(details)
    assert cross.system == system
    assert cross.cpu == cpu
    assert cross.cpu_family == cpu_family
    assert cross.binary_format == binary_format
    assert f'{cross.interpreter_tag}-{cross.abi_tag}-{cross.platform_tag}' == tag


def test_cross_build_python_version():
    cross = mesonpy._cross.CrossBuild(build_details())
    assert cross.version_info == (3, 14)
    assert cross.python_version == '3.14.8'
    assert not cross.is_free_threaded
    assert not cross.is_pypy
    assert cross.python_include_dir == '/opt/host-python/include/python3.14'
    assert cross.python_pkgconfig_path == '/opt/host-python/lib/pkgconfig'


def test_cross_build_missing_platform():
    details = build_details('', multiarch=None, extension_suffix='.so')
    with pytest.raises(ValueError, match='empty "platform"'):
        mesonpy._cross.CrossBuild(details)


def test_cross_file(tmp_path):
    host_requires = tmp_path / 'host-requires'
    (host_requires / 'numpy' / '_core' / 'lib' / 'pkgconfig').mkdir(parents=True)
    (host_requires / 'numpy' / '_core' / 'lib' / 'pkgconfig' / 'numpy.pc').write_text('')
    toolchain = {
        'CC': 'aarch64-linux-gnu-gcc',
        'CXX': 'aarch64-linux-gnu-g++ -std=c++17',
        'AR': 'aarch64-linux-gnu-ar',
        'CFLAGS': '-O2 -g',
        'LDFLAGS': '-Wl,-z,relro',
    }
    cross = mesonpy._cross.CrossBuild(build_details(), os.fspath(host_requires), toolchain)
    details_path = tmp_path / 'build-details.json'
    cross.write_build_details(details_path)
    assert json.loads(details_path.read_text())['platform'] == 'linux-aarch64'

    text = cross.cross_file(details_path)
    assert "c = 'aarch64-linux-gnu-gcc'" in text
    assert "cpp = ['aarch64-linux-gnu-g++', '-std=c++17']" in text
    assert "ar = 'aarch64-linux-gnu-ar'" in text
    assert "pkg-config = 'pkg-config'" in text
    assert "system = 'linux'" in text
    assert "cpu_family = 'aarch64'" in text
    assert "endian = 'little'" in text
    assert 'needs_exe_wrapper = true' in text
    assert f"python.build_config = '{details_path}'" in text
    assert f"host_requires_directory = '{host_requires}'" in text
    assert f"pkg_config_path = ['{host_requires / 'numpy' / '_core' / 'lib' / 'pkgconfig'}']" in text
    assert "c_args = ['-O2', '-g']" in text
    assert "c_link_args = ['-Wl,-z,relro']" in text
    assert "cpp_link_args = ['-Wl,-z,relro']" in text


def test_cross_file_relative_base_prefix(tmp_path):
    cross = mesonpy._cross.CrossBuild(build_details(base_prefix='../..'))
    with pytest.raises(ValueError, match='base_prefix'):
        cross.write_build_details(tmp_path / 'build-details.json')


def test_wheel_tag_cross():
    cross = mesonpy._cross.CrossBuild(build_details())
    builder = wheel_builder_test_factory({'platlib': ['extension.cpython-314-aarch64-linux-gnu.so']}, pure=False)
    builder._cross = cross
    assert str(builder.tag) == 'cp314-cp314-linux_aarch64'


def test_wheel_tag_cross_no_extension_modules():
    cross = mesonpy._cross.CrossBuild(build_details())
    builder = wheel_builder_test_factory({'mesonpy-libs': ['libfoo.so']}, pure=False)
    builder._cross = cross
    assert str(builder.tag) == 'py3-none-linux_aarch64'


def test_wheel_tag_cross_stable_abi():
    cross = mesonpy._cross.CrossBuild(build_details())
    builder = wheel_builder_test_factory({'platlib': ['extension.abi3.so']}, pure=False, limited_api=True)
    builder._cross = cross
    assert str(builder.tag) == 'cp314-abi3-linux_aarch64'


def test_wheel_tag_cross_stable_abi_free_threaded():
    details = build_details(version=(3, 15, 0), flags=['t'], extension_suffix='.cpython-315t-aarch64-linux-gnu.so')
    cross = mesonpy._cross.CrossBuild(details)
    builder = wheel_builder_test_factory({'platlib': ['extension.abi3t.so']}, pure=False, limited_api=True)
    builder._cross = cross
    assert str(builder.tag) == 'cp315-abi3.abi3t-linux_aarch64'


def test_wheel_tag_cross_stable_abi_bad():
    cross = mesonpy._cross.CrossBuild(build_details())
    builder = wheel_builder_test_factory({'platlib': ['extension.cpython-314-aarch64-linux-gnu.so']},
                                         pure=False, limited_api=True)
    builder._cross = cross
    with pytest.raises(mesonpy.BuildError, match='tagged for a specific Python version'):
        str(builder.tag)


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux only')
def test_get_build_requires_for_cross_build_wheel(monkeypatch):
    monkeypatch.setenv('NINJA', 'ninja')
    monkeypatch.setattr(shutil, 'which', lambda name: None)
    # ELF host on an ELF build platform: patchelf is needed
    assert mesonpy.get_build_requires_for_cross_build_wheel({}, build_details()) == ['patchelf >= 0.11.0']
    # Mach-O host: rpath fixing is skipped, patchelf is not needed
    macos = build_details('macosx-11.0-arm64', multiarch='darwin', extension_suffix='.cpython-314-darwin.so')
    assert mesonpy.get_build_requires_for_cross_build_wheel({}, macos) == []
    # same as get_requires_for_build_wheel when the host matches the build platform
    assert mesonpy.get_requires_for_build_wheel({}) == ['patchelf >= 0.11.0']


def test_get_build_requires_for_cross_build_wheel_requires_build_details():
    with pytest.raises(SystemExit):
        mesonpy.get_build_requires_for_cross_build_wheel({}, None)


def test_cross_build_wheel_requires_build_details(package_pure, tmp_path):
    with pytest.raises(SystemExit):
        mesonpy.cross_build_wheel(os.fspath(tmp_path), build_details=None)


# A "cross" build where the host is the same machine we are running on:
# exercises the whole code path (cross file, python.build_config, tags
# derived from build-details.json) without needing a cross compiler, by
# describing the running interpreter with a synthesized build-details.json.
def running_interpreter_build_details():
    version = sys.version_info
    multiarch = getattr(sys.implementation, '_multiarch', None)
    return {
        'schema_version': '1.0',
        'base_prefix': sys.base_prefix,
        'base_interpreter': sys.executable,
        'platform': sysconfig.get_platform(),
        'language': {
            'version': f'{version.major}.{version.minor}',
            'version_info': {
                'major': version.major, 'minor': version.minor, 'micro': version.micro,
                'releaselevel': version.releaselevel, 'serial': version.serial,
            },
        },
        'implementation': {
            'name': sys.implementation.name,
            'cache_tag': sys.implementation.cache_tag,
            **({'_multiarch': multiarch} if multiarch else {}),
        },
        'abi': {
            'flags': list(sys.abiflags),
            'extension_suffix': sysconfig.get_config_var('EXT_SUFFIX'),
            'stable_abi_suffix': '.abi3.so',
        },
        'libpython': {
            'dynamic': os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY') or ''),
            'link_extensions': False,
        },
        'c_api': {
            'headers': sysconfig.get_path('include'),
            'pkgconfig_path': os.path.join(sysconfig.get_config_var('LIBDIR'), 'pkgconfig'),
        },
    }


@pytest.mark.skipif(MESON_VERSION < (1, 10, 0), reason='python.build_config requires meson 1.10')
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux only')
@pytest.mark.skipif(sys.implementation.name != 'cpython', reason='CPython only')
@pytest.mark.skipif(not shutil.which('pkg-config'), reason='requires pkg-config')
def test_cross_build_wheel_same_machine(package_purelib_and_platlib, tmp_path):
    details = running_interpreter_build_details()
    toolchain = {'CC': os.environ.get('CC', 'cc'), 'STRIP': os.environ.get('STRIP', 'strip')}
    with chdir(package_purelib_and_platlib), in_git_repo_context():
        name = mesonpy.cross_build_wheel(os.fspath(tmp_path), build_details=details, toolchain=toolchain)
    expected = f'{mesonpy._tags.get_interpreter_tag()}-{mesonpy._tags.get_abi_tag()}-{mesonpy._tags.get_platform_tag()}'
    assert name == f'purelib_and_platlib-1.0.0-{expected}.whl'
    assert (tmp_path / name).exists()
