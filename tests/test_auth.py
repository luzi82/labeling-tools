import unittest

from fastapi.testclient import TestClient

from main import app, config


class AuthenticationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app, follow_redirects=False)

    def test_home_redirects_to_login_when_unauthenticated(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/login")

    def test_home_is_available_after_login(self) -> None:
        login_response = self.client.post(
            "/login", data={"password": config["password"]}
        )
        response = self.client.get("/")

        self.assertEqual(login_response.status_code, 303)
        self.assertEqual(login_response.headers["location"], "/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('class="grid-space"', response.text)


if __name__ == "__main__":
    unittest.main()