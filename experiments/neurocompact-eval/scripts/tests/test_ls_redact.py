"""Offline tests: secret redaction. Secret-looking strings are assembled from parts so that no
literal credential-shaped token sits in this file."""
from __future__ import annotations

import unittest
from collections import Counter

import ls_fixtures as F  # noqa: F401
from ls_redact import format_report, redact, redact_session

A = "a1B2c3D4" * 5


class RedactTests(unittest.TestCase):
    def kinds(self, text):
        out, c = redact(text)
        return out, dict(c)

    def test_provider_keys(self):
        samples = {
            "sk_ant": "s" + "k-ant-" + "api03-" + A,
            "sk_or": "s" + "k-or-" + "v1-" + A,
            "sk": "s" + "k-" + A,
            "github": "gh" + "p_" + A[:36],
            "gitlab": "glp" + "at-" + A[:20],
            "aws_key": "AK" + "IA" + "ABCDEFGHIJKLMNOP",
        }
        for kind, tok in samples.items():
            out, c = self.kinds(f"before {tok} after")
            self.assertNotIn(tok, out, kind)
            self.assertEqual(c.get(kind), 1, (kind, c))
            self.assertIn("before [REDACTED:" + kind + "] after", out)

    def test_ant_key_is_not_double_counted_as_plain_sk(self):
        _, c = self.kinds("s" + "k-ant-" + A)
        self.assertEqual(c, {"sk_ant": 1})

    def test_private_key_block_multiline(self):
        pem = ("-----BEGIN RSA PRIVATE" + " KEY-----\nMIIE" + A + "\nabc\n-----END RSA PRIVATE" + " KEY-----")
        out, c = self.kinds(f"x\n{pem}\ny")
        self.assertEqual(out, "x\n[REDACTED:private_key]\ny")
        self.assertEqual(c["private_key"], 1)

    def test_unterminated_private_key_is_cut_to_end(self):
        out, c = self.kinds("head\n-----BEGIN PRIVATE" + " KEY-----\nMIIabc\nmore")
        self.assertEqual(out, "head\n[REDACTED:private_key]")

    def test_assignments_keep_the_name_and_hide_secret_looking_values(self):
        vals = {"Zx9fQ2mK7pLw4RtY8vNb": "api_key", "eW91ci1zZWNyZXQtdmFsdWU9": "auth_token",
                "s3Cr3t-V4lue-9xQ2mK7pLw": "DB_PASSWORD", "a1B2c3D4e5F6g7H8": "client_secret",
                "django-insecure-8f3k2j9d0s1l": "SECRET_KEY", "Tr0ub4dor3xKq9zLmQ": "passwd"}
        text = " ".join(f"{k}={v}" for v, k in vals.items()) + ' accessToken: "Qw3rTy9uIo5pAs7dFg"'
        out, c = self.kinds(text)
        for v in list(vals) + ["Qw3rTy9uIo5pAs7dFg"]:
            self.assertNotIn(v, out)
        for k in vals.values():
            self.assertIn(f"{k}=[REDACTED:assign]", out)
        self.assertEqual(c["assign"], 7)

    def test_json_and_quoted_assignment_forms(self):
        out, c = self.kinds('{"api_key": "Zx9fQ2mK7pLw4RtY8vNb", "model": "claude-sonnet-5-5"}')
        self.assertEqual(out, '{"api_key": "[REDACTED:assign]", "model": "claude-sonnet-5-5"}')
        out, c = self.kinds("export AUTH_TOKEN='Zx9fQ2mK7pLw4RtY8vNb'")
        self.assertEqual(out, "export AUTH_TOKEN='[REDACTED:assign]'")

    def test_assignment_false_positives_are_left_alone(self):
        # the log line that used to become `tokenizer=...tiktoken:[REDACTED:assign]`
        fixtures = [
            "tokenizer=tiktoken:cl100k_base", "tokenizer: tiktoken:cl100k_base", "tiktoken:cl100k_base",
            "max_tokens=100000 token=None password=$PASSWORD secret=${SECRET} api_key=os.environ['K']",
            "token: str password= author=anthropic-engineering-team",
            "secret=claude-sonnet-5-5-20260101 token_model=claude-haiku-5-5",   # model ids are not secrets
            "auth_file=/home/user/.config/gh/hosts.yml passwd=/etc/shadow.backup.file",
            "api_key_env=ANTHROPIC_API_KEY_FOR_TESTS token_count=1234567890123",
            "token=https://example.com/v1/some/long/path/here auth=Bearer",
            "auth_mode=oauth_device_flow_v2 credentials_file=creds/service-account.json",
            "session_token_id=3f2b8c1e-aaaa-bbbb-cccc-1234567890ab",
            "password=short tokenizer_name=BertWordPieceTokenizerFast",
            "getAuthToken=makeAuthTokenFromConfig", "bearer_prefix=Authorization-Header-Value",
            "secret_name=my-production-database-credentials",
        ]
        for text in fixtures:
            skipped = Counter()
            out, c = redact(text, skipped)
            self.assertEqual(out, text, text)
            self.assertEqual(c, {}, text)

    def test_key_must_have_a_secret_word_segment(self):
        v = "Zx9fQ2mK7pLw4RtY8vNb"
        for key in ("checksum", "tokenizer", "author", "monkey_id", "keyboard", "session_id", "my_key"):
            t = f"{key}={v}"
            self.assertEqual(redact(t)[0], t, key)
        for key in ("token", "TOKEN", "my_secret", "dbPassword", "apiKey", "api-key", "x.auth", "bearer",
                    "private_key", "AWS_ACCESS_KEY", "credential"):
            self.assertEqual(redact(f"{key}={v}")[0], f"{key}=[REDACTED:assign]", key)

    def test_skipped_counter_reports_matched_but_not_secret_like(self):
        sk = Counter()
        redact("secret=claude-sonnet-5-5-20260101 tokenizer=tiktoken:cl100k_base", sk)
        self.assertEqual(sk["assign_not_secret_like"], 1)       # `tokenizer` has no secret key word at all

    def test_per_pattern_counts_in_report(self):
        text = ("s" + "k-ant-" + A + " api_key=Zx9fQ2mK7pLw4RtY8vNb tokenizer=tiktoken:cl100k_base "
                "gh" + "p_" + A[:36])
        _, c = redact(text)
        self.assertEqual(dict(c), {"sk_ant": 1, "assign": 1, "github": 1})
        rep = format_report(c)
        self.assertIn("assign=1", rep)
        self.assertIn("sk_ant=1", rep)
        self.assertIn("total 3", rep)

    def test_aws_secret_jwt_bearer_slack_google(self):
        jwt = "ey" + "J" + "h" * 10 + ".ey" + "J" + "p" * 10 + "." + "s" * 12
        text = ("aws_secret_access_key = " + "A" * 40 + "\n" + jwt + "\nAuthorization: Bearer " + A + "\n"
                + "xo" + "xb-" + "1234567890-abcdefghij" + "\n" + "AI" + "za" + "B" * 35)
        out, c = self.kinds(text)
        for k in ("aws_secret", "jwt", "bearer", "slack", "google_api"):
            self.assertEqual(c.get(k), 1, (k, c))
        self.assertNotIn(A, out)

    def test_clean_text_unchanged_and_idempotent(self):
        t = "def f(x):\n    return x + 1  # nothing secret, sk is a prefix, token count 3"
        self.assertEqual(redact(t), (t, {}))
        once, _ = redact("password=Zx9fQ2mK7pLw4RtY8vNb")
        twice, c2 = redact(once)
        self.assertEqual(once, twice)
        self.assertEqual(c2, {})

    def test_session_report(self):
        s = F.make_session(n=4, questions=False)
        s["messages"][1]["text"] = "use s" + "k-or-v1-" + A
        s["compactions"] = [{"at_idx": 2, "original_summary": "token=Zx9fQ2mK7pLw4RtY8vNb"}]
        s["title"] = "t"
        tot = redact_session(s)
        self.assertEqual(tot["sk_or"], 1)
        self.assertEqual(tot["assign"], 1)
        self.assertEqual(s["meta"]["redaction"]["total"], 2)
        self.assertIn("sk_or=1", format_report(tot))
        self.assertNotIn("abcdef999999", s["compactions"][0]["original_summary"])


if __name__ == "__main__":
    unittest.main()
