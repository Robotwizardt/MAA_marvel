from pathlib import Path

import shutil
import sys

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import jsonc
except ModuleNotFoundError as e:
    raise ImportError(
        "Missing dependency 'json-with-comments' (imported as 'jsonc').\n"
        f"Install it with:\n  {sys.executable} -m pip install json-with-comments\n"
        "Or add it to your project's requirements."
    ) from e

try:
    from tools.configure import configure_ocr_model
except ModuleNotFoundError:
    from configure import configure_ocr_model

from agent.runtime.task_cache import migrate_runtime_task_cache


def get_dotnet_platform_tag(os_name: str, arch: str) -> str:
    platforms = {
        ("win", "x86_64"): "win-x64",
        ("win", "aarch64"): "win-arm64",
        ("macos", "x86_64"): "osx-x64",
        ("macos", "aarch64"): "osx-arm64",
        ("linux", "x86_64"): "linux-x64",
        ("linux", "aarch64"): "linux-arm64",
    }
    try:
        return platforms[(os_name, arch)]
    except KeyError as error:
        raise ValueError(f"Unsupported platform: {os_name}/{arch}") from error


def install_deps(
    source_root: Path,
    destination: Path,
    os_name: str,
    arch: str,
) -> None:
    deps = source_root / "deps"
    if not (deps / "bin").exists():
        raise FileNotFoundError('Please download MaaFramework to "deps" first.')

    if os_name == "android":
        shutil.copytree(deps / "bin", destination, dirs_exist_ok=True)
        shutil.copytree(
            deps / "share" / "MaaAgentBinary",
            destination / "MaaAgentBinary",
            dirs_exist_ok=True,
        )
        return

    platform_tag = get_dotnet_platform_tag(os_name, arch)
    shutil.copytree(
        deps / "bin",
        destination / "runtimes" / platform_tag / "native",
        ignore=shutil.ignore_patterns(
            "*MaaDbgControlUnit*",
            "*MaaThriftControlUnit*",
            "*MaaRpc*",
            "*MaaHttp*",
            "plugins",
            "*.node",
            "*MaaPiCli*",
        ),
        dirs_exist_ok=True,
    )
    shutil.copytree(
        deps / "share" / "MaaAgentBinary",
        destination / "libs" / "MaaAgentBinary",
        dirs_exist_ok=True,
    )
    shutil.copytree(
        deps / "bin" / "plugins",
        destination / "plugins" / platform_tag,
        dirs_exist_ok=True,
    )


def install_project_files(
    source_root: Path,
    destination: Path,
    release_version: str,
    os_name: str | None = None,
) -> None:
    """Copy project-owned release files without local development output."""
    shutil.copytree(
        source_root / "assets" / "resource",
        destination / "resource",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        dirs_exist_ok=True,
    )
    shutil.copy2(source_root / "assets" / "interface.json", destination)
    shutil.copytree(
        source_root / "assets" / "tasks",
        destination / "tasks",
        dirs_exist_ok=True,
    )

    interface_path = destination / "interface.json"
    with interface_path.open("r", encoding="utf-8") as stream:
        interface = jsonc.load(stream)
    interface["version"] = release_version
    agent_bundle = source_root / "agent_dist" / "MAA_marvel_agent"
    if os_name != "android" and agent_bundle.exists():
        shutil.copytree(
            agent_bundle,
            destination / "agent_runtime",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            dirs_exist_ok=True,
        )
        executable = "MAA_marvel_agent.exe" if os_name == "win" else "MAA_marvel_agent"
        interface["agent"] = {
            "child_exec": f"./agent_runtime/{executable}",
            "child_args": [],
        }
    with interface_path.open("w", encoding="utf-8") as stream:
        jsonc.dump(interface, stream, ensure_ascii=False, indent=4)

    shutil.copytree(
        source_root / "agent",
        destination / "agent",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        dirs_exist_ok=True,
    )
    shutil.copy2(source_root / "README.md", destination)
    shutil.copy2(source_root / "LICENSE", destination)
    migrate_runtime_task_cache(destination)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv if argv is None else argv
    if len(args) < 4:
        print("Usage: python install.py <version> <os> <arch>")
        print("Example: python install.py v1.0.0 win x86_64")
        return 1

    release_version, os_name, arch = args[1:4]
    source_root = Path(__file__).parent.parent.resolve()
    destination = source_root / "install"

    try:
        configure_ocr_model()
        install_deps(source_root, destination, os_name, arch)
        install_project_files(source_root, destination, release_version, os_name)
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return 1

    print(f"Install to {destination} successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
