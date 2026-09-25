import csv
import importlib.util
import re
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient
from PIL import Image


MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"
spec = importlib.util.spec_from_file_location("versus_mobile_main", MAIN_PATH)
assert spec is not None and spec.loader is not None
versus_mobile = importlib.util.module_from_spec(spec)
sys.modules["versus_mobile_main"] = versus_mobile
spec.loader.exec_module(versus_mobile)

app = versus_mobile.app
build_runtime = versus_mobile.build_runtime
config = versus_mobile.config
initialize_output = versus_mobile.initialize_output
parse_startup_arguments = versus_mobile.parse_startup_arguments
read_rows = versus_mobile.read_rows


class ComparisonTests(unittest.TestCase):
  def setUp(self) -> None:
    self.client = TestClient(app, follow_redirects=False)

  def login(self) -> None:
    response = self.client.post("/login", data={"password": config["password"]})
    self.assertEqual(response.status_code, 303)

  def use_runtime(self, runtime) -> None:
    previous_runtime = app.state.runtime
    app.state.runtime = runtime
    self.addCleanup(setattr, app.state, "runtime", previous_runtime)

  def write_images(self, folder: Path, count: int) -> list[Path]:
    folder.mkdir()
    paths: list[Path] = []
    for index in range(count):
      image_path = folder / f"image-{index}.png"
      Image.new("RGB", (8, 8), color=(index, 0, 0)).save(image_path)
      paths.append(image_path.resolve())
    return paths

  def pair_from(self, page: str) -> tuple[str, str] | None:
    if 'data-state="done"' in page:
      return None
    match = re.search(r'data-image-a="([^"]*)" data-image-b="([^"]*)"', page)
    self.assertIsNotNone(match)
    assert match is not None
    return match.group(1), match.group(2)

  def open_runtime(self, image_count: int):
    directory = TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    root = Path(directory.name)
    images = self.write_images(root / "images", image_count)
    output = root / "output"
    runtime = build_runtime(image_folder=root / "images", output_folder=output)
    initialize_output(runtime)
    self.use_runtime(runtime)
    self.login()
    return runtime, images

  def test_home_redirects_when_unauthenticated(self) -> None:
    response = self.client.get("/")
    self.assertEqual(response.status_code, 303)
    self.assertEqual(response.headers["location"], "/login")

  def test_submitted_images_are_not_shown_again(self) -> None:
    runtime, images = self.open_runtime(4)
    first = self.pair_from(self.client.get("/").text)
    assert first is not None
    submit = self.client.post("/comparisons", data={"image_a": first[0], "image_b": first[1], "result": "A>B"})
    second_page = self.client.get("/")
    second = self.pair_from(second_page.text)

    self.assertEqual(submit.status_code, 303)
    self.assertIsNotNone(second)
    assert second is not None
    self.assertTrue(set(first).isdisjoint(second))
    self.assertEqual(set(first) | set(second), {str(image) for image in images})
    self.assertIn('class="slot" name="result" type="submit" value="A&gt;B"', second_page.text)
    self.assertIn('class="tie" name="result" type="submit" value="A≈B">≈</button>', second_page.text)
    self.assertIn('class="slot" name="result" type="submit" value="A&lt;B"', second_page.text)
    self.assertNotIn("<figcaption>", second_page.text)
    self.assertEqual(read_rows(runtime), [{"image_a": first[0], "image_b": first[1], "result": "A>B"}])

  def test_undo_restores_the_last_pair(self) -> None:
    runtime, _images = self.open_runtime(4)
    first = self.pair_from(self.client.get("/").text)
    assert first is not None
    self.client.post("/comparisons", data={"image_a": first[0], "image_b": first[1], "result": "A>B"})
    second = self.pair_from(self.client.get("/").text)
    assert second is not None
    self.client.post("/comparisons", data={"image_a": second[0], "image_b": second[1], "result": "A≈B"})

    done = self.client.get("/")
    undo = self.client.post("/comparisons/undo")
    restored_page = self.client.get("/")

    self.assertIn('data-state="done"', done.text)
    self.assertEqual(undo.status_code, 303)
    self.assertEqual(self.pair_from(restored_page.text), second)
    self.assertNotIn(first[0], restored_page.text)
    self.assertEqual(read_rows(runtime), [{"image_a": first[0], "image_b": first[1], "result": "A>B"}])

  def test_done_when_fewer_than_two_images_remain(self) -> None:
    _runtime, images = self.open_runtime(1)
    page = self.client.get("/")

    self.assertEqual(page.status_code, 200)
    self.assertIn('data-state="done"', page.text)
    self.assertIn("No pairs left.", page.text)
    self.assertNotIn('id="undo-button"', page.text)
    self.assertNotIn(str(images[0]), page.text)

  def test_refresh_keeps_the_current_pair(self) -> None:
    self.open_runtime(4)
    first = self.client.get("/")
    second = self.client.get("/")
    self.assertEqual(self.pair_from(first.text), self.pair_from(second.text))
    self.assertIn('id="undo-button" type="submit" disabled', first.text)

  def test_startup_arguments_create_output_folder(self) -> None:
    with TemporaryDirectory() as directory:
      root = Path(directory)
      images = root / "images"
      images.mkdir()
      output = root / "nested" / "output"
      arguments = parse_startup_arguments(["--image-folder", str(images), "--output-folder", str(output)])
      self.assertEqual(arguments.image_folder, images.resolve())
      self.assertEqual(arguments.output_folder, output.resolve())
      self.assertTrue(output.is_dir())

  def test_image_is_served_only_from_the_source_folder(self) -> None:
    _runtime, images = self.open_runtime(2)
    available = self.client.get("/image", params={"image_path": str(images[0])})
    outside = images[0].parent.parent / "outside.png"
    Image.new("RGB", (8, 8)).save(outside)
    missing = self.client.get("/image", params={"image_path": str(outside)})

    self.assertEqual(available.status_code, 200)
    self.assertEqual(missing.status_code, 404)
    with (versus_mobile.result_path(app.state.runtime)).open(newline="", encoding="utf-8") as csv_file:
      self.assertEqual(tuple(csv.DictReader(csv_file).fieldnames or ()), ("image_a", "image_b", "result"))


if __name__ == "__main__":
  unittest.main()
