# Security Policy

## Supported Versions

Security fixes target the default branch of this repository. If you are running
an older commit, please upgrade before reporting version-specific issues.

## Reporting a Vulnerability

**Do not** open a public GitHub issue for security vulnerabilities.

Please report vulnerabilities privately by email to:

**lava@infervisor.ai**

Include as much of the following as you can:

- A clear description of the issue and its impact
- Steps to reproduce, or a proof of concept
- Affected components (`plowc`, `plowrt`, runtime kernels, deps, etc.)
- Your assessment of severity, if known

We aim to acknowledge reports within a few business days. After triage we will
coordinate disclosure timing with you. Please give us a reasonable window to
investigate and ship a fix before any public discussion.

## Scope

In scope: plow compiler, runtime, packet ABI handling, serving surface, and
build/scripts shipped in this repository.

Out of scope (unless plow is clearly at fault): third-party model weights,
external GPU drivers/toolchains, or issues that require physical or privileged
host access beyond normal inference deployment.
