import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient
from PIL import Image

from classification.main import app, config, parse_startup_arguments


class ClassificationTests(unittest.TestCase):
  def setUp(self) -> None:
    self.client = TestClient(app, follow_redirects=False)

  def login(self) -> None:
    response = self.client.post("/login", data={"password": config["password"]})
    self.assertEqual(response.status_code, 303)

  def write_config(self, working_folder: Path, labels: list[str], image_folder: Path) -> None:
    (working_folder / "classification_config.yaml").write_text(
      "label_list:\n" + "".join(f"  - {label}\n" for label in labels)
      + f"image_folder_path_list:\n  - {image_folder}\n",
      encoding="utf-8",
    )

  def use_working_folder(self, working_folder: Path):
    previous_working_folder = app.state.working_folder
    app.state.working_folder = working_folder
    return previous_working_folder

  def test_home_redirects_when_unauthenticated(self) -> None:
    response = self.client.get("/")
    self.assertEqual(response.status_code, 303)
    self.assertEqual(response.headers["location"], "/login")

  def test_valid_login_displays_completion_when_no_images_exist(self) -> None:
    with TemporaryDirectory() as directory:
      working_folder = Path(directory)
      image_folder = working_folder / "images"
      image_folder.mkdir()
      self.write_config(working_folder, ["cat"], image_folder)
      previous_working_folder = self.use_working_folder(working_folder)
      try:
        self.login()
        response = self.client.get("/")
      finally:
        app.state.working_folder = previous_working_folder

    self.assertEqual(response.status_code, 200)
    self.assertIn("All images are classified.", response.text)

  def test_discovers_subfolder_image_and_writes_one_classification(self) -> None:
    with TemporaryDirectory() as directory:
      working_folder = Path(directory)
      image_folder = working_folder / "images"
      nested_folder = image_folder / "nested"
      nested_folder.mkdir(parents=True)
      image_path = nested_folder / "sample.png"
      Image.new("RGB", (8, 8)).save(image_path)
      self.write_config(working_folder, ["first", "second"], image_folder)
      previous_working_folder = self.use_working_folder(working_folder)
      try:
        self.login()
        home_response = self.client.get("/")
        response = self.client.post(
          "/classifications", json={"image_path": str(image_path), "label": "second"}
        )
        duplicate_response = self.client.post(
          "/classifications", json={"image_path": str(image_path), "label": "first"}
        )
      finally:
        app.state.working_folder = previous_working_folder

      with (working_folder / "classification.csv").open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    self.assertEqual(home_response.status_code, 200)
    self.assertIn('data-label="first"', home_response.text)
    self.assertEqual(response.status_code, 200)
    self.assertEqual(duplicate_response.status_code, 409)
    self.assertEqual(rows, [{"image_path": str(image_path.resolve()), "label": "second"}])

  def test_duplicate_csv_path_is_rejected(self) -> None:
    with TemporaryDirectory() as directory:
      working_folder = Path(directory)
      image_folder = working_folder / "images"
      image_folder.mkdir()
      image_path = image_folder / "sample.jpg"
      Image.new("RGB", (8, 8)).save(image_path)
      self.write_config(working_folder, ["label"], image_folder)
      (working_folder / "classification.csv").write_text(
        f"image_path,label\n{image_path.resolve()},label\n{image_path.resolve()},label\n",
        encoding="utf-8",
      )
      previous_working_folder = self.use_working_folder(working_folder)
      try:
        self.login()
        response = self.client.get("/")
      finally:
        app.state.working_folder = previous_working_folder

    self.assertEqual(response.status_code, 422)
    self.assertIn("Duplicate image_path", response.text)

  def test_more_than_nine_labels_is_rejected(self) -> None:
    with TemporaryDirectory() as directory:
      working_folder = Path(directory)
      image_folder = working_folder / "images"
      image_folder.mkdir()
      self.write_config(working_folder, [f"label-{index}" for index in range(10)], image_folder)
      previous_working_folder = self.use_working_folder(working_folder)
      try:
        self.login()
        response = self.client.get("/")
      finally:
        app.state.working_folder = previous_working_folder

    self.assertEqual(response.status_code, 422)
    self.assertIn("between 1 and 9", response.text)

  def test_working_folder_argument_resolves_directory(self) -> None:
    with TemporaryDirectory() as directory:
      working_folder = Path(directory) / "new" / "workspace"
      arguments = parse_startup_arguments(["--working-folder", str(working_folder)])
      self.assertTrue(working_folder.is_dir())
      self.assertEqual(arguments.working_folder, working_folder.resolve())


if __name__ == "__main__":
  unittest.main()