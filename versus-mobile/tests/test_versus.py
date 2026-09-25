import contextlib
import csv
import importlib.util
import io
import re
import sys
import time
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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

  def excluded_images_of(self, runtime) -> list[str]:
    with (runtime.output_folder / "excluded.csv").open(newline="", encoding="utf-8") as csv_file:
      return [row["image"] for row in csv.DictReader(csv_file)]

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

  def test_port_defaults_to_8000(self) -> None:
    with TemporaryDirectory() as directory:
      root = Path(directory)
      images = root / "images"
      images.mkdir()
      arguments = parse_startup_arguments([
        "--image-folder", str(images),
        "--output-folder", str(root / "output"),
      ])
      self.assertEqual(arguments.port, 8000)

  def test_port_argument_is_accepted(self) -> None:
    with TemporaryDirectory() as directory:
      root = Path(directory)
      images = root / "images"
      images.mkdir()
      arguments = parse_startup_arguments([
        "--image-folder", str(images),
        "--output-folder", str(root / "output"),
        "--port", "9001",
      ])
      self.assertEqual(arguments.port, 9001)

  def test_port_rejects_values_outside_1_to_65535(self) -> None:
    with TemporaryDirectory() as directory:
      root = Path(directory)
      images = root / "images"
      images.mkdir()
      base = ["--image-folder", str(images), "--output-folder", str(root / "output"), "--port"]
      for bad in ("0", "-1", "65536", "abc"):
        with self.subTest(port=bad):
          with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
              parse_startup_arguments([*base, bad])
          self.assertEqual(raised.exception.code, 2)

  def test_main_passes_port_to_uvicorn(self) -> None:
    with TemporaryDirectory() as directory:
      root = Path(directory)
      images = root / "images"
      images.mkdir()
      previous = app.state.runtime
      self.addCleanup(setattr, app.state, "runtime", previous)
      with patch.object(versus_mobile.uvicorn, "run") as run:
        versus_mobile.main([
          "--image-folder", str(images),
          "--output-folder", str(root / "output"),
          "--port", "9001",
        ])
      run.assert_called_once_with(app, host="0.0.0.0", port=9001)

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

  def test_excluding_one_image_keeps_the_other_in_place(self) -> None:
    runtime, _images = self.open_runtime(4)
    first_page = self.client.get("/")
    first = self.pair_from(first_page.text)
    assert first is not None
    self.assertIn('class="exclude exclude-a" method="post"', first_page.text)
    self.assertIn(">A</button>", first_page.text)
    self.assertIn('class="exclude exclude-b" method="post"', first_page.text)
    self.assertIn(">B</button>", first_page.text)

    excluded = self.client.post("/exclusions", data={"image": first[0]})
    second = self.pair_from(self.client.get("/").text)

    self.assertEqual(excluded.status_code, 303)
    self.assertIsNotNone(second)
    assert second is not None
    self.assertEqual(second[1], first[1])
    self.assertNotEqual(second[0], first[0])
    self.assertEqual(read_rows(runtime), [])
    self.assertEqual(self.excluded_images_of(runtime), [first[0]])

  def test_excluded_image_is_not_compared_later(self) -> None:
    runtime, _images = self.open_runtime(5)
    first = self.pair_from(self.client.get("/").text)
    assert first is not None
    self.client.post("/exclusions", data={"image": first[0]})
    second = self.pair_from(self.client.get("/").text)
    assert second is not None
    self.client.post("/comparisons", data={"image_a": second[0], "image_b": second[1], "result": "A>B"})
    third = self.pair_from(self.client.get("/").text)

    self.assertIsNotNone(third)
    assert third is not None
    self.assertNotIn(first[0], second)
    self.assertNotIn(first[0], third)
    self.assertTrue(all(first[0] not in (row["image_a"], row["image_b"]) for row in read_rows(runtime)))

  def test_excluding_the_last_opponent_leaves_the_other_image_unused(self) -> None:
    runtime, _images = self.open_runtime(2)
    first = self.pair_from(self.client.get("/").text)
    assert first is not None
    self.client.post("/exclusions", data={"image": first[1]})
    page = self.client.get("/")

    self.assertIn('data-state="done"', page.text)
    self.assertIn("No pairs left.", page.text)
    self.assertEqual(self.excluded_images_of(runtime), [first[1]])

  def test_cannot_exclude_an_image_outside_the_current_pair(self) -> None:
    _runtime, images = self.open_runtime(4)
    pair = self.pair_from(self.client.get("/").text)
    assert pair is not None
    outsider = next(str(image) for image in images if str(image) not in pair)
    response = self.client.post("/exclusions", data={"image": outsider})

    self.assertEqual(response.status_code, 409)
    self.assertEqual(self.excluded_images_of(app.state.runtime), [])

  def test_undo_reverts_the_latest_exclusion_before_an_earlier_comparison(self) -> None:
    runtime, _images = self.open_runtime(4)
    first = self.pair_from(self.client.get("/").text)
    assert first is not None
    self.client.post("/comparisons", data={"image_a": first[0], "image_b": first[1], "result": "A>B"})
    second = self.pair_from(self.client.get("/").text)
    assert second is not None
    self.client.post("/exclusions", data={"image": second[0]})

    undo_exclusion = self.client.post("/comparisons/undo")
    restored = self.pair_from(self.client.get("/").text)

    self.assertEqual(undo_exclusion.status_code, 303)
    self.assertEqual(restored, second)
    self.assertEqual(read_rows(runtime), [{"image_a": first[0], "image_b": first[1], "result": "A>B"}])
    self.assertEqual(self.excluded_images_of(runtime), [])

    undo_comparison = self.client.post("/comparisons/undo")
    restored_again = self.pair_from(self.client.get("/").text)

    self.assertEqual(undo_comparison.status_code, 303)
    self.assertEqual(restored_again, first)
    self.assertEqual(read_rows(runtime), [])

  def test_undo_restores_an_exclusion_when_nothing_has_been_compared(self) -> None:
    runtime, _images = self.open_runtime(4)
    first_page = self.client.get("/")
    first = self.pair_from(first_page.text)
    assert first is not None
    self.assertIn('id="undo-button" type="submit" disabled', first_page.text)
    self.client.post("/exclusions", data={"image": first[0]})
    excluded_page = self.client.get("/")

    self.assertIn('id="undo-button" type="submit"', excluded_page.text)
    self.assertNotIn('id="undo-button" type="submit" disabled', excluded_page.text)

    undo = self.client.post("/comparisons/undo")
    restored = self.pair_from(self.client.get("/").text)

    self.assertEqual(undo.status_code, 303)
    self.assertEqual(restored, first)
    self.assertEqual(read_rows(runtime), [])
    self.assertEqual(self.excluded_images_of(runtime), [])

  def test_legacy_comparisons_csv_becomes_history(self) -> None:
    directory = TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    root = Path(directory.name)
    images = self.write_images(root / "images", 4)
    output = root / "output"
    output.mkdir()
    with (output / "comparisons.csv").open("w", encoding="utf-8", newline="") as csv_file:
      writer = csv.DictWriter(csv_file, fieldnames=("image_a", "image_b", "result"))
      writer.writeheader()
      writer.writerow({"image_a": str(images[0]), "image_b": str(images[1]), "result": "A>B"})
      writer.writerow({"image_a": str(images[2]), "image_b": str(images[3]), "result": "A<B"})
    runtime = build_runtime(image_folder=root / "images", output_folder=output)
    initialize_output(runtime)
    self.use_runtime(runtime)
    self.login()

    with (output / "history.csv").open(newline="", encoding="utf-8") as csv_file:
      self.assertEqual(list(csv.DictReader(csv_file)), [
        {"action": "comparison", "image_a": str(images[0]), "image_b": str(images[1]), "result": "A>B"},
        {"action": "comparison", "image_a": str(images[2]), "image_b": str(images[3]), "result": "A<B"},
      ])
    self.assertEqual(self.excluded_images_of(runtime), [])

    undo = self.client.post("/comparisons/undo")
    restored = self.pair_from(self.client.get("/").text)

    self.assertEqual(undo.status_code, 303)
    self.assertEqual(restored, (str(images[2]), str(images[3])))
    self.assertEqual(read_rows(runtime), [{"image_a": str(images[0]), "image_b": str(images[1]), "result": "A>B"}])

  def test_discover_dated_images_checks_only_selected_directory(self) -> None:
    with TemporaryDirectory() as directory:
      root = Path(directory)
      chosen = root / "2026-09" / "2026-09-25"
      chosen.mkdir(parents=True)
      current = chosen / "2026-09-25-7.png"
      current.touch()
      previous = root / "2026-09" / "2026-09-24"
      previous.mkdir()
      (previous / "2026-09-24-9.png").touch()
      (chosen / "other.txt").touch()
      with patch.object(Path, "rglob", side_effect=AssertionError("no recursive scan")):
        found = versus_mobile.discover_dated_images(root, date(2026, 9, 25))
      self.assertEqual(found, {str(current.resolve())})

  def test_refresh_runtime_includes_each_date_since_last_success(self) -> None:
    with TemporaryDirectory() as directory:
      root = Path(directory)
      images = root / "images"
      images.mkdir()
      original = images / "old.png"
      original.touch()
      runtime = build_runtime(image_folder=images, output_folder=root / "output")
      new_paths = []
      for day in ("2026-09-24", "2026-09-25", "2026-09-26"):
        folder = images / day[:7] / day
        folder.mkdir(parents=True)
        image = folder / f"{day}-1.png"
        image.touch()
        new_paths.append(str(image.resolve()))
      with patch.object(Path, "rglob", side_effect=AssertionError("no repeat full crawl")):
        refreshed = versus_mobile.refresh_runtime(runtime, date(2026, 9, 24), date(2026, 9, 26))
      self.assertEqual(refreshed.source_images, frozenset({str(original.resolve()), *new_paths}))
      self.assertEqual(runtime.source_images, frozenset({str(original.resolve())}))
      self.assertIsNot(refreshed, runtime)

  def test_refresh_swaps_runtime_and_preserves_current_pair(self) -> None:
    with TemporaryDirectory() as directory:
      root = Path(directory)
      images = self.write_images(root / "images", 2)
      runtime = build_runtime(image_folder=root / "images", output_folder=root / "output")
      initialize_output(runtime)
      self.use_runtime(runtime)
      self.login()
      before = self.pair_from(self.client.get("/").text)
      day = date(2026, 9, 25)
      folder = runtime.image_folder / "2026-09" / "2026-09-25"
      folder.mkdir(parents=True)
      new_images = [folder / f"2026-09-25-{seed}.png" for seed in (1, 2)]
      for image in new_images:
        image.touch()
      app.state.runtime = versus_mobile.refresh_runtime(runtime, day, day)
      after = self.pair_from(self.client.get("/").text)
      self.assertEqual(before, after)
      self.assertTrue({str(path.resolve()) for path in new_images} <= app.state.runtime.source_images)

  def test_lifespan_refreshes_new_images_without_manual_trigger(self) -> None:
    with TemporaryDirectory() as directory:
      root = Path(directory)
      images = root / "images"
      images.mkdir()
      runtime = build_runtime(image_folder=images, output_folder=root / "output")
      initialize_output(runtime)
      previous_runtime = app.state.runtime
      previous_date = app.state.refresh_date
      self.addCleanup(setattr, app.state, "runtime", previous_runtime)
      self.addCleanup(setattr, app.state, "refresh_date", previous_date)
      app.state.runtime = runtime
      app.state.refresh_date = date(2026, 9, 25)
      with patch.object(versus_mobile, "IMAGE_REFRESH_INTERVAL_SECONDS", 0.01), \
           patch.object(versus_mobile, "hk_today", return_value=date(2026, 9, 25)):
        with TestClient(app):
          folder = images / "2026-09" / "2026-09-25"
          folder.mkdir(parents=True)
          image = folder / "2026-09-25-1.png"
          image.touch()
          deadline = time.monotonic() + 2
          while str(image.resolve()) not in app.state.runtime.source_images and time.monotonic() < deadline:
            time.sleep(0.01)
          self.assertIn(str(image.resolve()), app.state.runtime.source_images)
          self.assertIn(str(image.resolve()), versus_mobile.unused_images(app.state.runtime, []))

if __name__ == "__main__":
  unittest.main()
