CREATE TABLE IF NOT EXISTS cbrain_durable_reviews (
    organisation_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    intent_digest TEXT NOT NULL,
    required_role TEXT NOT NULL,
    state TEXT NOT NULL,
    lease_owner TEXT,
    lease_expires_at TIMESTAMPTZ,
    actor_id TEXT,
    approved_at TIMESTAMPTZ,
    approval_expires_at TIMESTAMPTZ,
    consumed_at TIMESTAMPTZ,
    frozen_reason TEXT,
    version INTEGER NOT NULL,
    event_log JSONB NOT NULL DEFAULT '[]'::jsonb,

    PRIMARY KEY (organisation_id, request_id),

    CHECK (length(organisation_id) > 0),
    CHECK (length(request_id) > 0),
    CHECK (length(intent_digest) > 0),
    CHECK (version >= 1),
    CHECK (
        state IN (
            'PARKED',
            'LEASED',
            'APPROVED',
            'CONSUMED',
            'APPROVAL_EXPIRED',
            'FROZEN'
        )
    )
);

COMMENT ON TABLE cbrain_durable_reviews IS
  'Durable REVIEW parking. Lease expiry returns PARKED; it is not proof a write never happened.';
