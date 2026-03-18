import React, { useState, useEffect } from 'react';

const AdminPanel = () => {
    const [tenantName, setTenantName] = useState('');
    const [rpm, setRpm] = useState(60);
    const [createdTenant, setCreatedTenant] = useState(null);
    const [generatedKey, setGeneratedKey] = useState(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);
    const [tenantsList, setTenantsList] = useState([]);
    
    const fetchTenants = async () => {
        try {
            const endpoint = '/v1/admin/tenants';
            const res = await fetch(endpoint, {
                headers: {
                    'Authorization': 'Bearer super-secret-admin-token'
                }
            });
            if (res.ok) {
                const data = await res.json();
                setTenantsList(data);
            }
        } catch (err) {
            console.error("Failed to fetch tenants", err);
        }
    };

    useEffect(() => {
        fetchTenants();
    }, []);

    const handleCreateTenant = async () => {
        setLoading(true);
        setError(null);
        setCreatedTenant(null);
        setGeneratedKey(null);

        try {
            const endpoint = '/v1/admin/tenants';
            
            console.log(`Sending POST request to ${endpoint}`);
            const res = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer super-secret-admin-token'
                },
                body: JSON.stringify({
                    name: tenantName || 'New Tenant',
                    quotas: {
                        requests_per_minute: parseInt(rpm, 10)
                    }
                })
            });

            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            setCreatedTenant(data);
            fetchTenants();
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleGenerateKey = async () => {
        if (!createdTenant) return;
        setLoading(true);
        try {
            const endpoint = `/v1/admin/tenants/${createdTenant.id}/keys`;

            const res = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Authorization': 'Bearer super-secret-admin-token'
                }
            });
            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            setGeneratedKey(data.key);
            fetchTenants();
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ padding: '20px', background: '#e9ecef', borderBottom: '1px solid #dee2e6' }}>
            <h3 style={{ marginTop: 0 }}>Admin Control Plane</h3>
            <p style={{ fontSize: '0.9em', color: '#6c757d' }}>Simulate Admin actions to create a tenant and provision an API Key.</p>
            
            <div style={{ display: 'flex', gap: '10px', marginBottom: '10px', alignItems: 'center' }}>
                <input 
                    type="text" 
                    placeholder="Tenant Name" 
                    value={tenantName} 
                    onChange={e => setTenantName(e.target.value)} 
                    style={{ padding: '5px' }}
                />
                <label>
                    Req/Min (RPM):
                    <input 
                        type="number" 
                        value={rpm} 
                        onChange={e => setRpm(e.target.value)} 
                        style={{ padding: '5px', width: '60px', marginLeft: '5px' }}
                    />
                </label>
                <button onClick={handleCreateTenant} disabled={loading} style={{ padding: '5px 10px' }}>
                    1. Create Tenant
                </button>
            </div>

            {error && <div style={{ color: 'red', marginBottom: '10px' }}>Error: {error}</div>}

            {createdTenant && (
                <div style={{ background: '#fff', padding: '10px', borderRadius: '4px', marginBottom: '10px' }}>
                    <strong>Tenant Created:</strong> {createdTenant.name} <br/>
                    <small>ID: {createdTenant.id}</small><br/>
                    <button onClick={handleGenerateKey} disabled={loading || generatedKey} style={{ marginTop: '10px', padding: '5px 10px' }}>
                        2. Generate API Key
                    </button>
                </div>
            )}

            {generatedKey && (
                <div style={{ background: '#d4edda', color: '#155724', padding: '10px', borderRadius: '4px', border: '1px solid #c3e6cb' }}>
                    <strong>Generated API Key:</strong> <code style={{ userSelect: 'all', background: 'transparent' }}>{generatedKey}</code>
                    <p style={{ margin: '5px 0 0 0', fontSize: '0.9em' }}>Copy this key and paste it into the API Key input below to start chatting as this tenant. (Note: Keys are redacted in the list below for security).</p>
                </div>
            )}

            <h4 style={{ marginTop: '20px' }}>Existing Tenants</h4>
            {tenantsList.length === 0 ? (
                <p style={{ fontSize: '0.9em', color: '#6c757d' }}>No tenants found.</p>
            ) : (
                <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '10px', background: 'white' }}>
                    <thead>
                        <tr style={{ background: '#f8f9fa', borderBottom: '2px solid #dee2e6' }}>
                            <th style={{ padding: '8px', textAlign: 'left' }}>Tenant Name</th>
                            <th style={{ padding: '8px', textAlign: 'left' }}>ID</th>
                            <th style={{ padding: '8px', textAlign: 'left' }}>RPM Quota</th>
                            <th style={{ padding: '8px', textAlign: 'left' }}>API Keys (Redacted)</th>
                            <th style={{ padding: '8px', textAlign: 'left' }}>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {tenantsList.map(t => (
                            <tr key={t.id} style={{ borderBottom: '1px solid #dee2e6' }}>
                                <td style={{ padding: '8px' }}><strong>{t.name}</strong></td>
                                <td style={{ padding: '8px', fontSize: '0.85em', color: '#6c757d' }}>{t.id}</td>
                                <td style={{ padding: '8px' }}>{t.quotas?.requests_per_minute || 'N/A'}</td>
                                <td style={{ padding: '8px', fontSize: '0.9em' }}>
                                    {t.api_keys && t.api_keys.length > 0 ? (
                                        <ul style={{ margin: 0, paddingLeft: '20px' }}>
                                            {t.api_keys.map((k, i) => (
                                                <li key={i} style={{ color: k.is_active ? 'green' : 'red' }}>
                                                    <code>{k.prefix}</code> {k.is_active ? '(Active)' : '(Revoked)'}
                                                </li>
                                            ))}
                                        </ul>
                                    ) : (
                                        <span style={{ color: '#6c757d' }}>No keys</span>
                                    )}
                                </td>
                                <td style={{ padding: '8px' }}>
                                    <button 
                                        onClick={async () => {
                                            setCreatedTenant(t);
                                            await handleGenerateKey();
                                        }}
                                        disabled={loading}
                                        style={{ padding: '4px 8px', fontSize: '0.85em' }}
                                    >
                                        + New Key
                                    </button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
        </div>
    );
};

export default AdminPanel;