# Grafana Monitoring Service - Security Documentation

## Overview

This document outlines security considerations and best practices for the Grafana monitoring service in the YTrader trading system.

## Authentication

### Default Credentials

**CRITICAL**: The default Grafana admin credentials (`admin/admin`) are for development only. **MUST be changed in production**.

### Changing Admin Credentials

1. Update `.env` file:
   ```bash
   GRAFANA_ADMIN_USER=your_secure_username
   GRAFANA_ADMIN_PASSWORD=your_strong_password
   ```

2. Restart Grafana service:
   ```bash
   docker compose restart grafana
   ```

3. Verify new credentials work:
   ```bash
   curl -u your_secure_username:your_strong_password http://localhost:4700/api/user
   ```

### Password Requirements

- Minimum 8 characters (recommended: 16+)
- Mix of uppercase, lowercase, numbers, and special characters
- Not based on dictionary words
- Unique password (not reused from other services)

## Network Security

### Port Access

- **Grafana UI Port**: 4700 (non-standard port)
- **Internal Port**: 3000 (container internal, not exposed)

### Firewall Configuration

**Production Recommendations**:

1. **Restrict External Access**:
   ```bash
   # Allow only from specific IP ranges
   sudo ufw allow from 10.0.0.0/8 to any port 4700
   sudo ufw allow from 192.168.0.0/16 to any port 4700
   ```

2. **Use VPN or SSH Tunnel**:
   - Access Grafana via VPN connection
   - Or use SSH port forwarding:
     ```bash
     ssh -L 4700:localhost:4700 user@server
     ```

3. **Reverse Proxy with HTTPS**:
   - Use nginx/traefik as reverse proxy
   - Enable HTTPS/TLS encryption
   - Configure SSL certificates (Let's Encrypt recommended)

### Docker Network Isolation

Grafana runs in `ytrader-network` Docker network:
- Isolated from host network
- Only accessible to other services in the same network
- External access only via port mapping (4700)

## Data Source Credentials

### PostgreSQL Read-Only User

The `grafana_monitor` user has **read-only** access:

**Permissions**:
- `SELECT` on required tables only
- `USAGE` on `public` schema
- `CONNECT` to `ytrader` database

**Restrictions**:
- No `INSERT`, `UPDATE`, `DELETE` permissions
- No `CREATE`, `ALTER`, `DROP` permissions (DDL)
- No access to other databases
- No superuser privileges

**Credential Management**:
- Stored in `.env` file (not committed to git)
- Rotate periodically (recommended: every 90 days)
- Use strong passwords (16+ characters)

### RabbitMQ Management API

**Access**:
- Uses existing RabbitMQ credentials from `.env`
- Basic authentication via HTTP headers
- Read-only access to queue metrics

**Security**:
- Credentials stored in `.env` file
- Rotate periodically
- Consider separate read-only RabbitMQ user for Grafana

## Data Privacy

### Sensitive Data in Dashboards

**Trading Data**:
- Trading signals, orders, and execution events may contain sensitive trading information
- Access should be restricted to authorized personnel only
- Consider data retention policies for dashboard queries

**Model Information**:
- Model versions and quality metrics may reveal trading strategies
- Limit access to model development and operations teams

### Data Retention

- Dashboard queries limit results (100-200 records)
- Time range filters reduce data exposure
- Consider implementing data retention policies for historical data

## Container Security

### Base Image

- Uses official `grafana/grafana:10.4.0` image
- Regularly update to latest stable version
- Monitor security advisories for Grafana

### Container Configuration

**Non-Root User**:
- Grafana runs as non-root user (`grafana`, UID 472)
- Reduces risk of privilege escalation

**Read-Only Volumes**:
- Provisioning files mounted as read-only
- Dashboard files mounted as read-only
- Data volume is writable (for Grafana internal data)

**Health Checks**:
- Health check endpoint: `/api/health`
- Monitors container health
- Automatic restart on failure

## Credential Management Best Practices

### Environment Variables

1. **Never commit `.env` to git**:
   - `.env` is in `.gitignore`
   - Use `env.example` for documentation only

2. **Use secrets management**:
   - Consider Docker secrets or external secrets manager
   - Rotate credentials regularly

3. **Separate credentials per environment**:
   - Different credentials for dev/staging/production
   - Never reuse production credentials in development

### Credential Rotation

**Recommended Schedule**:
- Admin credentials: Every 90 days
- Database credentials: Every 180 days
- Service API keys: Every 90 days

**Rotation Process**:
1. Generate new credentials
2. Update `.env` file
3. Update database user password (for PostgreSQL)
4. Restart Grafana service
5. Verify connectivity
6. Remove old credentials from `.env`

## Access Control

### User Management

**Current Setup**: Single admin user

**Future Enhancements** (if needed):
- Multiple users with role-based access
- LDAP/OAuth integration
- API key authentication for programmatic access

### Dashboard Access

- All dashboards accessible to authenticated users
- Consider dashboard-level permissions for sensitive data
- Implement audit logging for dashboard access

## Monitoring and Auditing

### Access Logs

Grafana logs all access attempts:
- View logs: `docker compose logs grafana`
- Monitor for unauthorized access attempts
- Set up alerts for suspicious activity

### Health Monitoring

- Health endpoint: `/api/health`
- Monitor via external monitoring tools
- Set up alerts for service downtime

## Incident Response

### Security Breach Procedures

1. **Immediate Actions**:
   - Change all credentials immediately
   - Review access logs for unauthorized access
   - Isolate affected systems if necessary

2. **Investigation**:
   - Review Grafana logs
   - Check database access logs
   - Identify scope of breach

3. **Remediation**:
   - Rotate all credentials
   - Update security configurations
   - Patch vulnerabilities

4. **Documentation**:
   - Document incident
   - Update security procedures
   - Conduct post-incident review

## Compliance

### Data Protection

- Ensure compliance with data protection regulations (GDPR, etc.)
- Implement data retention policies
- Provide data export capabilities if required

### Audit Requirements

- Maintain access logs
- Document credential changes
- Regular security reviews

## Security Checklist

### Initial Setup

- [ ] Change default admin credentials
- [ ] Configure firewall rules
- [ ] Set up HTTPS (if external access)
- [ ] Verify read-only database user permissions
- [ ] Review and restrict network access

### Ongoing Maintenance

- [ ] Rotate credentials regularly
- [ ] Update Grafana to latest stable version
- [ ] Review access logs monthly
- [ ] Monitor security advisories
- [ ] Conduct security audits annually

## Additional Resources

- [Grafana Security Documentation](https://grafana.com/docs/grafana/latest/setup-grafana/configure-security/)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)

