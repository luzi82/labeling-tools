import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient
from PIL import Image

from classification.main import (
  HUMAN_CORRECTED,
  HUMAN_LABELED,
  HUMAN_SCREENED,
  NOT_HUMAN_CHECKED,
  app,
  build_runtime,
  config,
  initialize_output,
  parse_startup_arguments,
  read_result_rows,
)


class ClassificationTests(unittest.TestCase):
  def setUp(self) -> None:
    self.client = TestClient(app, follow_redirects=False)

  def login(self) -> None:
    response = self.client.post("/login", data={"password": config["password"]})
    self.assertEqual(response.status_code, 303)

  def write_meta(self, directory: Path, labels: list[str]) -> Path:
    meta_yaml = directory / "meta.yaml"
    meta_yaml.write_text("label_list:\n" + "".join(f"  - {label}\n" for label in labels), encoding="utf-8")
    return meta_yaml

  def use_runtime(self, runtime):
    previous_runtime = app.state.runtime
    app.state.runtime = runtime
    return previous_runtime

  def write_model_csv(self, path: Path, rows: list[tuple[Path, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
      writer = csv.DictWriter(csv_file, fieldnames=("image_path", "label"))
      writer.writeheader()
      for image_path, label in rows:
        writer.writerow({"image_path": str(image_path.resolve()), "label": label})

  def test_home_redirects_when_unauthenticated(self) -> None:
    response = self.client.get("/")
    self.assertEqual(response.status_code, 303)
    self.assertEqual(response.headers["location"], "/login")

  def test_initial_mode_labels_image_and_undo_restores_it(self) -> None:
    with TemporaryDirectory() as directory:
      root = Path(directory)
      images = root / "images"
      images.mkdir()
      image_path = images / "sample.png"
      next_image_path = images / "next.png"
      Image.new("RGB", (8, 8)).save(image_path)
      Image.new("RGB", (8, 8)).save(next_image_path)
      meta_yaml = self.write_meta(root, ["first", "second"])
      output = root / "output"
      runtime = build_runtime(input_csv=None, image_folder=images, meta_yaml=meta_yaml, output_folder=output)
      initialize_output(runtime)
      previous_runtime = self.use_runtime(runtime)
      try:
        self.login()
        home_response = self.client.get("/")
        create_response = self.client.post("/classifications", json={"image_path": str(image_path), "label": "second"})
        undo_response = self.client.post("/classifications/undo")
        restored_response = self.client.get("/", params={"image_path": str(image_path.resolve())})
      finally:
        app.state.runtime = previous_runtime

      rows = read_result_rows(runtime)

    self.assertEqual(home_response.status_code, 200)
    self.assertIn("Initial image labeling", home_response.text)
    self.assertIn('data-label="first"', home_response.text)
    self.assertEqual(create_response.json()["human_checked_state"], HUMAN_LABELED)
    self.assertEqual(undo_response.json(), {
      "action": "create",
      "changed_count": 1,
      "restored_image_path": str(image_path.resolve()),
    })
    self.assertEqual(restored_response.status_code, 200)
    self.assertIn(str(image_path.resolve()), restored_response.text)
    self.assertEqual(rows, [])

  def test_initial_mode_can_review_past_labels_by_label(self) -> None:
    with TemporaryDirectory() as directory:
      root = Path(directory)
      images = root / "images"
      images.mkdir()
      first_image = images / "first.png"
      second_image = images / "second.png"
      Image.new("RGB", (8, 8)).save(first_image)
      Image.new("RGB", (8, 8)).save(second_image)
      meta_yaml = self.write_meta(root, ["first", "second"])
      runtime = build_runtime(input_csv=None, image_folder=images, meta_yaml=meta_yaml, output_folder=root / "output")
      initialize_output(runtime)
      previous_runtime = self.use_runtime(runtime)
      try:
        self.login()
        self.client.post("/classifications", json={"image_path": str(first_image), "label": "first"})
        self.client.post("/classifications", json={"image_path": str(second_image), "label": "second"})
        correction_response = self.client.patch("/classifications", json={"image_path": str(first_image), "label": "second"})
        moved_items_response = self.client.get("/review/items", params={"label": "second"})
        undo_response = self.client.post("/classifications/undo")
        dashboard_response = self.client.get("/", params={"view": "review"})
        review_response = self.client.get("/review", params={"label": "first"})
        items_response = self.client.get("/review/items", params={"label": "first"})
      finally:
        app.state.runtime = previous_runtime

    self.assertTrue(correction_response.json()["updated"])
    self.assertIn({
      "image_path": str(first_image.resolve()),
      "label": "second",
      "human_checked_state": HUMAN_LABELED,
      "missing": False,
    }, moved_items_response.json()["items"])
    self.assertEqual(undo_response.json(), {"action": "update", "changed_count": 1})
    self.assertEqual(dashboard_response.status_code, 200)
    self.assertIn('href="/review?label=first"', dashboard_response.text)
    self.assertIn("1 images", dashboard_response.text)
    self.assertEqual(review_response.status_code, 200)
    self.assertEqual(items_response.json()["items"], [{
      "image_path": str(first_image.resolve()),
      "label": "first",
      "human_checked_state": HUMAN_LABELED,
      "missing": False,
    }])

  def test_model_output_is_materialized_and_dashboard_is_shown(self) -> None:
    with TemporaryDirectory() as directory:
      root = Path(directory)
      images = root / "images"
      images.mkdir()
      first_image = images / "first.png"
      second_image = images / "second.png"
      Image.new("RGB", (8, 8)).save(first_image)
      Image.new("RGB", (8, 8)).save(second_image)
      meta_yaml = self.write_meta(root, ["first", "second"])
      input_csv = root / "classification.csv"
      self.write_model_csv(input_csv, [(first_image, "first"), (second_image, "second")])
      runtime = build_runtime(input_csv=input_csv, image_folder=None, meta_yaml=meta_yaml, output_folder=root / "output")
      initialize_output(runtime)
      previous_runtime = self.use_runtime(runtime)
      try:
        self.login()
        response = self.client.get("/")
      finally:
        app.state.runtime = previous_runtime

      rows = read_result_rows(runtime)

    self.assertEqual(rows, [
      {"image_path": str(first_image.resolve()), "label": "first", "human_checked_state": NOT_HUMAN_CHECKED},
      {"image_path": str(second_image.resolve()), "label": "second", "human_checked_state": NOT_HUMAN_CHECKED},
    ])
    self.assertEqual(response.status_code, 200)
    self.assertIn("Model label verification", response.text)
    self.assertIn("Double-check corrected images", response.text)
    self.assertEqual(response.text.count("1 unchecked"), 2)

  def test_model_correction_moves_to_corrected_queue_and_undo_restores_state(self) -> None:
    with TemporaryDirectory() as directory:
      root = Path(directory)
      images = root / "images"
      images.mkdir()
      image_path = images / "sample.png"
      Image.new("RGB", (8, 8)).save(image_path)
      meta_yaml = self.write_meta(root, ["first", "second"])
      input_csv = root / "classification.csv"
      self.write_model_csv(input_csv, [(image_path, "first")])
      runtime = build_runtime(input_csv=input_csv, image_folder=None, meta_yaml=meta_yaml, output_folder=root / "output")
      initialize_output(runtime)
      previous_runtime = self.use_runtime(runtime)
      try:
        self.login()
        update_response = self.client.patch("/classifications", json={"image_path": str(image_path), "label": "second"})
        first_items = self.client.get("/review/items", params={"label": "first"})
        corrected_items = self.client.get("/review/items", params={"corrected_only": "true"})
        undo_response = self.client.post("/classifications/undo")
      finally:
        app.state.runtime = previous_runtime

      rows = read_result_rows(runtime)

    self.assertTrue(update_response.json()["updated"])
    self.assertEqual(first_items.json()["items"], [])
    self.assertEqual(corrected_items.json()["items"][0]["human_checked_state"], HUMAN_CORRECTED)
    self.assertEqual(undo_response.json(), {"action": "update", "changed_count": 1})
    self.assertEqual(rows[0]["label"], "first")
    self.assertEqual(rows[0]["human_checked_state"], NOT_HUMAN_CHECKED)

  def test_completing_model_label_screens_only_unmodified_rows_and_is_undoable(self) -> None:
    with TemporaryDirectory() as directory:
      root = Path(directory)
      images = root / "images"
      images.mkdir()
      first_image = images / "first.png"
      second_image = images / "second.png"
      Image.new("RGB", (8, 8)).save(first_image)
      Image.new("RGB", (8, 8)).save(second_image)
      meta_yaml = self.write_meta(root, ["first", "second"])
      input_csv = root / "classification.csv"
      self.write_model_csv(input_csv, [(first_image, "first"), (second_image, "first")])
      runtime = build_runtime(input_csv=input_csv, image_folder=None, meta_yaml=meta_yaml, output_folder=root / "output")
      initialize_output(runtime)
      previous_runtime = self.use_runtime(runtime)
      try:
        self.login()
        self.client.patch("/classifications", json={"image_path": str(first_image), "label": "second"})
        complete_response = self.client.post("/review/complete", params={"label": "first"})
        after_complete = read_result_rows(runtime)
        undo_response = self.client.post("/classifications/undo")
      finally:
        app.state.runtime = previous_runtime

      after_undo = read_result_rows(runtime)

    self.assertEqual(complete_response.json(), {"changed_count": 1})
    self.assertEqual(after_complete[0]["human_checked_state"], HUMAN_CORRECTED)
    self.assertEqual(after_complete[1]["human_checked_state"], HUMAN_SCREENED)
    self.assertEqual(undo_response.json(), {"action": "screen", "changed_count": 1})
    self.assertEqual(after_undo[1]["human_checked_state"], NOT_HUMAN_CHECKED)

  def test_output_rejects_changed_model_source_on_resume(self) -> None:
    with TemporaryDirectory() as directory:
      root = Path(directory)
      images = root / "images"
      images.mkdir()
      image_path = images / "sample.png"
      Image.new("RGB", (8, 8)).save(image_path)
      meta_yaml = self.write_meta(root, ["first", "second"])
      input_csv = root / "classification.csv"
      self.write_model_csv(input_csv, [(image_path, "first")])
      output = root / "output"
      runtime = build_runtime(input_csv=input_csv, image_folder=None, meta_yaml=meta_yaml, output_folder=output)
      initialize_output(runtime)
      self.write_model_csv(input_csv, [(image_path, "second")])
      changed_runtime = build_runtime(input_csv=input_csv, image_folder=None, meta_yaml=meta_yaml, output_folder=output)
      with self.assertRaisesRegex(ValueError, "does not match"):
        initialize_output(changed_runtime)

  def test_review_paginates_model_items(self) -> None:
    with TemporaryDirectory() as directory:
      root = Path(directory)
      images = root / "images"
      images.mkdir()
      image_paths = []
      for index in range(101):
        image_path = images / f"image-{index:03}.png"
        Image.new("RGB", (8, 8)).save(image_path)
        image_paths.append(image_path)
      meta_yaml = self.write_meta(root, ["first"])
      input_csv = root / "classification.csv"
      self.write_model_csv(input_csv, [(image_path, "first") for image_path in image_paths])
      runtime = build_runtime(input_csv=input_csv, image_folder=None, meta_yaml=meta_yaml, output_folder=root / "output")
      initialize_output(runtime)
      previous_runtime = self.use_runtime(runtime)
      try:
        self.login()
        first_page = self.client.get("/review/items", params={"label": "first"})
        second_page = self.client.get("/review/items", params={"label": "first", "offset": 100})
      finally:
        app.state.runtime = previous_runtime

    self.assertEqual(len(first_page.json()["items"]), 100)
    self.assertTrue(first_page.json()["has_more"])
    self.assertEqual(second_page.json()["items"][0]["image_path"], str(image_paths[100].resolve()))

  def test_startup_arguments_require_a_mode_and_resolve_paths(self) -> None:
    with TemporaryDirectory() as directory:
      root = Path(directory)
      images = root / "images"
      images.mkdir()
      meta_yaml = self.write_meta(root, ["first"])
      output = root / "nested" / "output"
      arguments = parse_startup_arguments(["--image-folder", str(images), "--meta-yaml", str(meta_yaml), "--output-folder", str(output)])
    self.assertEqual(arguments.image_folder, images.resolve())
    self.assertEqual(arguments.output_folder, output.resolve())


if __name__ == "__main__":
  unittest.main()
