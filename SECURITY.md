# Security policy

## Supported versions

Only the latest release on the `main` branch (currently v1.0.x) receives fixes.

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Use GitHub's private
vulnerability reporting:

<https://github.com/ycl-berserk/quarry-gpu/security/advisories/new>

Include the affected version/commit, a minimal reproduction (scene size, command line
or Python snippet), and what you observed. We will acknowledge within a few days and
coordinate a fix and disclosure.

## Scope

quarry is a computational library: the main risk surface is untrusted geometry or query
input causing out-of-bounds reads/writes, excessive memory allocation, or a CUDA fault
that takes down the process. Reports about GPU driver bugs are welcome but should also
be filed with the vendor.
