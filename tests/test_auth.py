import unittest
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

from fastapi.testclient import TestClient
from PIL import Image

from main import app, config, parse_startup_arguments


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
        self.assertIn('class="menu-panel"', response.text)
        self.assertIn('action="/logout"', response.text)
        self.assertIn('class="grid-space"', response.text)
        self.assertIn('id="add-images-folder-button"', response.text)
        self.assertIn("Add images folder", response.text)
        self.assertIn('id="add-images-folder-dialog"', response.text)
        self.assertIn('for="image-folder-path">Path</label>', response.text)
        self.assertIn('for="image-label">Label</label>', response.text)
        self.assertIn('class="grid-space"', response.text)
        self.assertIn('className = "folder-tile workspace-item"', response.text)
        self.assertIn("function panWorkspace", response.text)
        self.assertIn('textContent = "Import"', response.text)
        self.assertIn('className = "image-tile workspace-item"', response.text)
        self.assertIn('querySelectorAll(".workspace-item")', response.text)

    def test_working_folder_argument_resolves_directory(self) -> None:
        with TemporaryDirectory() as directory:
            working_folder = Path(directory) / "new" / "workspace"
            arguments = parse_startup_arguments(
                ["--working-folder", str(working_folder)]
            )

            self.assertTrue(working_folder.is_dir())
            self.assertEqual(arguments.working_folder, working_folder.resolve())

    def test_creating_image_folder_appends_csv_entry(self) -> None:
        with TemporaryDirectory() as directory:
            previous_working_folder = app.state.working_folder
            app.state.working_folder = Path(directory)
            try:
                self.client.post("/login", data={"password": config["password"]})
                response = self.client.post(
                    "/image-folders",
                    json={"label": "Training images", "path": "/images/train", "x": 32, "y": 32},
                )
            finally:
                app.state.working_folder = previous_working_folder

            self.assertEqual(response.status_code, 200)
            UUID(response.json()["uuid"])
            with (Path(directory) / "image_folders.csv").open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(rows, [{
            "uuid": response.json()["uuid"],
            "label": "Training images",
            "path": "/images/train",
            "x": "32.0",
            "y": "32.0",
        }])

    def test_listing_image_folders_reads_csv_entries(self) -> None:
        with TemporaryDirectory() as directory:
            working_folder = Path(directory)
            with (working_folder / "image_folders.csv").open("w", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file, fieldnames=("uuid", "label", "path", "x", "y")
                )
                writer.writeheader()
                writer.writerow({
                    "uuid": "9c0ccd87-3d32-4f3d-8a71-a261d12bc530",
                    "label": "Saved images",
                    "path": "/images/saved",
                    "x": "96",
                    "y": "144",
                })

            previous_working_folder = app.state.working_folder
            app.state.working_folder = working_folder
            try:
                self.client.post("/login", data={"password": config["password"]})
                response = self.client.get("/image-folders")
            finally:
                app.state.working_folder = previous_working_folder

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [{
            "uuid": "9c0ccd87-3d32-4f3d-8a71-a261d12bc530",
            "label": "Saved images",
            "path": "/images/saved",
            "x": 96.0,
            "y": 144.0,
        }])

    def test_importing_images_returns_ten_bounded_images(self) -> None:
        with TemporaryDirectory() as directory:
            working_folder = Path(directory)
            image_folder = working_folder / "source-images"
            image_folder.mkdir()
            for index in range(12):
                Image.new("RGB", (800, 400), color=(index, 0, 0)).save(
                    image_folder / f"image-{index}.png"
                )
            folder_id = "9c0ccd87-3d32-4f3d-8a71-a261d12bc530"
            with (working_folder / "image_folders.csv").open("w", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file, fieldnames=("uuid", "label", "path", "x", "y")
                )
                writer.writeheader()
                writer.writerow({
                    "uuid": folder_id,
                    "label": "Source images",
                    "path": str(image_folder),
                    "x": "32",
                    "y": "32",
                })

            previous_working_folder = app.state.working_folder
            app.state.working_folder = working_folder
            try:
                self.client.post("/login", data={"password": config["password"]})
                response = self.client.post(f"/image-folders/{folder_id}/import")
            finally:
                app.state.working_folder = previous_working_folder

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 10)
        for image in response.json():
            self.assertLessEqual(image["width"], 384)
            self.assertLessEqual(image["height"], 384)
            self.assertLessEqual(abs(image["width"] * image["height"] - 65_536), 384)


if __name__ == "__main__":
    unittest.main()