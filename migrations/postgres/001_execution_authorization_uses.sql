CREATE TABLE IF NOT EXISTS cbrain_execution_authorization_uses (
    organisation_id TEXT NOT NULL,
    execution_authorization_id TEXT NOT NULL,
    authorization_digest TEXT NOT NULL,
    request_id TEXT NOT NULL,
    claimed_at TIMESTAMPTZ NOT NULL,

    PRIMARY KEY (
        organisation_id,
        execution_authorization_id
    ),

    UNIQUE (authorization_digest),

    CHECK (length(organisation_id) > 0),
    CHECK (length(execution_authorization_id) > 0),
    CHECK (
        authorization_digest
        ~ '^sha256:[0-9a-f]{64}$'
    ),
    CHECK (length(request_id) > 0)
);
