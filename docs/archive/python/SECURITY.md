# Security Policy

## Supported Platforms

This project supports:
- **Windows 11**
- **macOS** (latest)
- **Linux** (Fedora, Ubuntu, Linux Mint)

## Overview

Spatial Memory MCP Server stores and retrieves potentially sensitive information as vector embeddings. This document outlines security considerations and best practices.

## Threat Model

### What This System Handles

- Text content stored as memories (may contain sensitive information)
- Vector embeddings derived from content
- Metadata including timestamps, tags, and custom fields
- Namespace isolation for multi-tenant scenarios

### Trust Boundaries

1. **LLM Client**: Trusted to make appropriate tool calls
2. **MCP Protocol**: Transport layer (assumed secure)
3. **Storage Layer**: LanceDB files on local filesystem
4. **Embedding Service**: Local model or OpenAI API

## Security Features

### Input Validation

- **SQL Injection Prevention**: All database queries use parameterized inputs with pattern detection for common injection attempts (UNION, --, etc.)
- **UUID Validation**: Memory IDs must be valid UUIDs
- **Namespace Format**: Namespaces are validated for safe characters
- **Content Length Limits**: Maximum 100,000 characters per memory
- **Numeric Bounds**: Importance scores constrained to 0.0-1.0

### Data Isolation

- **Namespace Isolation**: Memories are segregated by namespace
- **No Cross-Namespace Queries**: Operations are scoped to a single namespace

### Error Handling

- **Sanitized Errors**: Error messages do not leak internal paths or sensitive data
- **Typed Exceptions**: Clear exception hierarchy for different failure modes

## Security Considerations

### Data at Rest

LanceDB stores data in local files without encryption by default.

**Recommendation**: Use filesystem-level encryption for sensitive data:
- **Windows 11**: BitLocker
- **macOS**: FileVault
- **Linux**: LUKS

**Recommendation**: Set restrictive permissions on the storage directory.

#### Linux (Ubuntu, Fedora, Linux Mint)

```bash
# Create directory with restrictive permissions
mkdir -p ~/.spatial-memory
chmod 700 ~/.spatial-memory

# Verify permissions (should show drwx------)
ls -la ~/.spatial-memory
```

#### macOS

```bash
# Create directory with restrictive permissions
mkdir -p ~/.spatial-memory
chmod 700 ~/.spatial-memory

# Verify permissions
ls -la ~/.spatial-memory
```

#### Windows 11 (PowerShell - Run as Administrator)

```powershell
# Create the directory
$memoryPath = "$env:USERPROFILE\.spatial-memory"
New-Item -ItemType Directory -Force -Path $memoryPath

# Remove inherited permissions and set owner-only access
$acl = Get-Acl $memoryPath
$acl.SetAccessRuleProtection($true, $false)
$acl.Access | ForEach-Object { $acl.RemoveAccessRule($_) } | Out-Null
$rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
    $env:USERNAME,
    "FullControl",
    "ContainerInherit,ObjectInherit",
    "None",
    "Allow"
)
$acl.SetAccessRule($rule)
Set-Acl $memoryPath $acl

# Verify (should show only your username with FullControl)
Get-Acl $memoryPath | Format-List
```

#### Windows 11 (GUI Alternative)

1. Navigate to `%USERPROFILE%\.spatial-memory` in File Explorer
2. Right-click → Properties → Security tab
3. Click "Advanced" → "Disable inheritance" → "Remove all inherited permissions"
4. Click "Add" → "Select a principal" → Enter your username
5. Check "Full control" → OK → Apply

### API Keys

- OpenAI API keys should be stored in environment variables, not in code
- **Never commit `.env` files** containing API keys
- Use a secrets manager for production deployments

#### Linux / macOS

```bash
# Add to ~/.bashrc, ~/.zshrc, or ~/.profile
export SPATIAL_MEMORY_OPENAI_API_KEY="sk-..."

# Reload shell
source ~/.bashrc  # or ~/.zshrc
```

#### Windows 11 (PowerShell)

```powershell
# Set for current user (persistent)
[Environment]::SetEnvironmentVariable(
    "SPATIAL_MEMORY_OPENAI_API_KEY",
    "sk-...",
    "User"
)

# Or use System Properties → Environment Variables GUI
```

### Memory Content

- Memories may contain sensitive information extracted from conversations
- Consider what data is appropriate to persist
- Implement data retention policies using the `decay` feature
- Use `forget` to remove sensitive memories when no longer needed

### Network Security

- Local embedding models (sentence-transformers) require no network access after initial download
- OpenAI embeddings send content to OpenAI's API
- **Recommendation**: Use local models for sensitive data

### Logging

- Debug logging may include memory content
- **Recommendation**: Use `INFO` or higher log level in production

```bash
SPATIAL_MEMORY_LOG_LEVEL=INFO
```

## Backup and Recovery

LanceDB stores data as files in the storage directory. Back up by copying this directory.

#### Linux / macOS

```bash
# Backup
cp -r ~/.spatial-memory ~/.spatial-memory-backup-$(date +%Y%m%d)

# Restore
cp -r ~/.spatial-memory-backup-20260129 ~/.spatial-memory
```

#### Windows 11 (PowerShell)

```powershell
# Backup
$date = Get-Date -Format "yyyyMMdd"
Copy-Item -Recurse "$env:USERPROFILE\.spatial-memory" "$env:USERPROFILE\.spatial-memory-backup-$date"

# Restore
Copy-Item -Recurse "$env:USERPROFILE\.spatial-memory-backup-20260129" "$env:USERPROFILE\.spatial-memory"
```

## Secure Configuration Checklist

- [ ] Storage directory has restrictive permissions
- [ ] API keys are in environment variables, not code
- [ ] `.env` file is in `.gitignore`
- [ ] Log level is `INFO` or higher in production
- [ ] Using local embeddings for sensitive data (or accepting OpenAI's privacy policy)
- [ ] Filesystem encryption enabled for storage directory
- [ ] Regular backups of memory database

## Vulnerability Reporting

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Use [GitHub's private vulnerability reporting](https://github.com/arman-tech/spatial-memory-mcp/security/advisories/new)
3. Or email security concerns to the repository maintainers (see GitHub profile)
4. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will respond within 48 hours and work with you to address the issue.

## Known Limitations

1. **No Authentication**: The MCP server does not implement authentication; it relies on the MCP client for access control
2. **No Encryption**: Data is not encrypted at rest by default
3. **Namespace Trust**: Namespace isolation is logical, not cryptographic
4. **Embedding Reversibility**: While difficult, embeddings may leak information about original content

## Security Updates

Security updates will be released as patch versions. Subscribe to repository releases for notifications:
- Watch the repository on GitHub
- Or use `gh repo subscribe arman-tech/spatial-memory-mcp --issues --pulls`
