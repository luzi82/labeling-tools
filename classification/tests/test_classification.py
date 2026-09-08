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
    self.assertIn('id="undo-button"', response.text)
    self.assertIn("undoLastClassification", response.text)

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

  def test_undo_removes_latest_classification_and_restores_its_image(self) -> None:
    with TemporaryDirectory() as directory:
      working_folder = Path(directory)
      image_folder = working_folder / "images"
      image_folder.mkdir()
      first_image = image_folder / "first.png"
      second_image = image_folder / "second.png"
      Image.new("RGB", (8, 8)).save(first_image)
      Image.new("RGB", (8, 8)).save(second_image)
      self.write_config(working_folder, ["label"], image_folder)
      previous_working_folder = self.use_working_folder(working_folder)
      try:
        self.login()
        self.client.post("/classifications", json={"image_path": str(first_image), "label": "label"})
        self.client.post("/classifications", json={"image_path": str(second_image), "label": "label"})
        undo_response = self.client.post("/classifications/undo")
        restored_response = self.client.get(
          "/", params={"image_path": str(second_image.resolve())}
        )
        redo_response = self.client.post(
          "/classifications", json={"image_path": str(second_image), "label": "label"}
        )
        next_image_response = self.client.get("/")
      finally:
        app.state.working_folder = previous_working_folder

      with (working_folder / "classification.csv").open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    self.assertEqual(undo_response.status_code, 200)
    self.assertEqual(undo_response.json()["image_path"], str(second_image.resolve()))
    self.assertEqual(restored_response.status_code, 200)
    self.assertIn(str(second_image.resolve()), restored_response.text)
    self.assertEqual(redo_response.status_code, 200)
    self.assertEqual(next_image_response.status_code, 200)
    self.assertIn("All images are classified.", next_image_response.text)
    self.assertEqual(rows, [
      {"image_path": str(first_image.resolve()), "label": "label"},
      {"image_path": str(second_image.resolve()), "label": "label"},
    ])

  def test_review_paginates_items_and_undo_restores_corrected_label(self) -> None:
    with TemporaryDirectory() as directory:
      working_folder = Path(directory)
      image_folder = working_folder / "images"
      image_folder.mkdir()
      labels = ["first", "second"]
      self.write_config(working_folder, labels, image_folder)
      image_paths = []
      for index in range(101):
        image_path = image_folder / f"image-{index:03}.png"
        Image.new("RGB", (8, 8)).save(image_path)
        image_paths.append(image_path)
      with (working_folder / "classification.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=("image_path", "label"))
        writer.writeheader()
        for image_path in image_paths:
          writer.writerow({"image_path": str(image_path.resolve()), "label": "first"})

      previous_working_folder = self.use_working_folder(working_folder)
      try:
        self.login()
        home_response = self.client.get("/")
        first_page = self.client.get("/review/items", params={"label": "first"})
        second_page = self.client.get("/review/items", params={"label": "first", "offset": 100})
        update_response = self.client.patch(
          "/classifications", json={"image_path": str(image_paths[0]), "label": "second"}
        )
        first_label_after_update = self.client.get("/review/items", params={"label": "first"})
        second_label_after_update = self.client.get("/review/items", params={"label": "second"})
        undo_response = self.client.post("/classifications/undo")
        second_label_after_undo = self.client.get("/review/items", params={"label": "second"})
      finally:
        app.state.working_folder = previous_working_folder

      with (working_folder / "classification.csv").open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    self.assertEqual(home_response.status_code, 200)
    self.assertIn('href="/review?label=first"', home_response.text)
    self.assertIn("first (101)", home_response.text)
    self.assertEqual(first_page.status_code, 200)
    self.assertEqual(len(first_page.json()["items"]), 100)
    self.assertTrue(first_page.json()["has_more"])
    self.assertEqual(second_page.json()["items"], [{
      "image_path": str(image_paths[100].resolve()), "label": "first", "missing": False,
    }])
    self.assertEqual(update_response.status_code, 200)
    self.assertTrue(update_response.json()["updated"])
    self.assertNotIn(str(image_paths[0].resolve()), {
      item["image_path"] for item in first_label_after_update.json()["items"]
    })
    self.assertEqual(second_label_after_update.json()["items"], [{
      "image_path": str(image_paths[0].resolve()), "label": "second", "missing": False,
    }])
    self.assertEqual(undo_response.status_code, 200)
    self.assertEqual(undo_response.json()["action"], "update")
    self.assertEqual(undo_response.json()["label"], "first")
    self.assertEqual(second_label_after_undo.json()["items"], [])
    self.assertEqual(sum(row["image_path"] == str(image_paths[0].resolve()) for row in rows), 1)
    self.assertEqual(rows[0]["label"], "first")

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