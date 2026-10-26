"""Offline tests: secret redaction. Secret-looking strings are assembled from parts so that no
literal credential-shaped token sits in this file."""
from __future__ import annotations

import unittest

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

    def test_assignments_keep_the_name_and_hide_the_value(self):
        text = 'password=hunter2xyz token: "abcdef123456" API_KEY=zzzzzzzz1 DB_PASSWORD = s3cretvalue'
        out, c = self.kinds(text)
        for v in ("hunter2xyz", "abcdef123456", "zzzzzzzz1", "s3cretvalue"):
            self.assertNotIn(v, out)
        self.assertIn("password=[REDACTED:assign]", out)
        self.assertEqual(c["assign"], 4)

    def test_assignment_false_positives_are_left_alone(self):
        text = ("max_tokens=100000 tokenizer=cl100k_base token=None password=$PASSWORD secret=${SECRET} "
                "api_key=os.environ['K'] token: str password=")
        out, c = self.kinds(text)
        self.assertEqual(out, text)
        self.assertEqual(c, {})

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
        once, _ = redact("password=hunter2xyz")
        twice, c2 = redact(once)
        self.assertEqual(once, twice)
        self.assertEqual(c2, {})

    def test_session_report(self):
        s = F.make_session(n=4, questions=False)
        s["messages"][1]["text"] = "use s" + "k-or-v1-" + A
        s["compactions"] = [{"at_idx": 2, "original_summary": "token=abcdef999999"}]
        s["title"] = "t"
        tot = redact_session(s)
        self.assertEqual(tot["sk_or"], 1)
        self.assertEqual(tot["assign"], 1)
        self.assertEqual(s["meta"]["redaction"]["total"], 2)
        self.assertIn("sk_or=1", format_report(tot))
        self.assertNotIn("abcdef999999", s["compactions"][0]["original_summary"])


if __name__ == "__main__":
    unittest.main()
