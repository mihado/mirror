use pretty_assertions::assert_eq;

use super::*;

#[test]
fn the_egress_profile_writes_the_ca_and_points_tools_at_it() {
    let profile = egress_profile("public CA", &["SERVICE_TOKEN".to_string()]).expect("profile");
    assert_eq!(
        profile,
        GuestProfile {
            name: "egress".into(),
            environment: HashMap::from([
                ("SSL_CERT_FILE".into(), "/tmp/mirror-egress-ca.pem".into()),
                ("GIT_SSL_CAINFO".into(), "/tmp/mirror-egress-ca.pem".into()),
                ("CURL_CA_BUNDLE".into(), "/tmp/mirror-egress-ca.pem".into()),
                (
                    "REQUESTS_CA_BUNDLE".into(),
                    "/tmp/mirror-egress-ca.pem".into()
                ),
                (
                    "NODE_EXTRA_CA_CERTS".into(),
                    "/tmp/mirror-egress-ca.pem".into()
                ),
                (
                    "CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE".into(),
                    "/tmp/mirror-egress-ca.pem".into()
                ),
                (
                    "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH".into(),
                    "/tmp/mirror-egress-ca.pem".into()
                ),
                ("SERVICE_TOKEN".into(), "mirror-managed".into()),
            ]),
            files: vec![GuestFile {
                path: "/tmp/mirror-egress-ca.pem".into(),
                mode: 0o644,
                contents: b"public CA".to_vec(),
            }],
            endpoints: Vec::new(),
        }
    );
}

#[test]
fn malformed_repeated_or_ca_placeholder_variables_are_refused() {
    assert_eq!(
        egress_profile("public CA", &["SERVICE-TOKEN".to_string()]),
        Err(
            "placeholder variable \"SERVICE-TOKEN\" is not a variable name, is repeated or names \
             a CA variable"
                .into()
        )
    );
    assert_eq!(
        egress_profile(
            "public CA",
            &["SERVICE_TOKEN".to_string(), "SERVICE_TOKEN".to_string()]
        ),
        Err(
            "placeholder variable \"SERVICE_TOKEN\" is not a variable name, is repeated or names \
             a CA variable"
                .into()
        )
    );
    assert_eq!(
        egress_profile("public CA", &["SSL_CERT_FILE".to_string()]),
        Err(
            "placeholder variable \"SSL_CERT_FILE\" is not a variable name, is repeated or names \
             a CA variable"
                .into()
        )
    );
}
