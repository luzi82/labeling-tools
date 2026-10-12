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
config = versus_mobile.config
parse_startup_arguments = versus_mobile.parse_startup_arguments
read_rows = versus_mobile.read_rows


class ComparisonTests(unittest.TestCase):
  def setUp(self) -> None:
    self.client = TestClient(app, follow_redirects=False)

  def login(self) -> None:
    response = self.client.post("/login", data={"password": config["password"]})
    self.assertEqual(response.status_code, 303)

  def use_state(self, catalog, workdir: Path) -> None:
    previous = (app.state.catalog, app.state.workdir, app.state.refresh_date)
    app.state.catalog = catalog
    app.state.workdir = workdir

    def restore() -> None:
      app.state.catalog, app.state.workdir, app.state.refresh_date = previous

    self.addCleanup(restore)

  def write_images(self, folder: Path, count: int) -> list[Path]:
    folder.mkdir(parents=True)
    paths: list[Path] = []
    for index in range(count):
      image_path = folder / f"image-{index}.png"
      Image.new("RGB", (8, 8), color=(index, 0, 0)).save(image_path)
      paths.append(image_path.resolve())
    return paths

  def excluded_images_of(self, store) -> list[str]:
    with (store.folder / "excluded.csv").open(newline="", encoding="utf-8") as csv_file:
      return [row["image"] for row in csv.DictReader(csv_file)]

  def pair_from(self, page: str) -> tuple[str, str] | None:
    if 'data-state="done"' in page:
      return None
    match = re.search(r'data-image-a="([^"]*)" data-image-b="([^"]*)"', page)
    self.assertIsNotNone(match)
    assert match is not None
    return match.group(1), match.group(2)

  def open_workspace(
    self,
    datasets: dict[str, int],
    *,
    versus: str | None = "main",
    select: list[str] | None = None,
  ):
    directory = TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    root = Path(directory.name)
    images_root = root / "images"
    images_root.mkdir()
    written: dict[str, list[Path]] = {}
    for name, count in datasets.items():
      written[name] = self.write_images(images_root / name, count)
    workdir = root / "work"
    workdir.mkdir()
    catalog = versus_mobile.build_catalog(images_root)
    self.use_state(catalog, workdir)
    self.login()
    if versus is not None:
      chosen = list(datasets) if select is None else select
      response = self.client.post("/new", data={"name": versus, "datasets": chosen})
      self.assertEqual(response.status_code, 303, response.headers.get("location"))
      self.assertEqual(response.headers["location"], f"/versus/{versus}")
    store = versus_mobile.make_store(catalog, versus, workdir / versus) if versus is not None else None
    return store, written, workdir, catalog

  def test_home_redirects_when_unauthenticated(self) -> None:
    response = self.client.get("/")
    self.assertEqual(response.status_code, 303)
    self.assertEqual(response.headers["location"], "/login")

  def test_submitted_images_are_not_shown_again(self) -> None:
    store, written, _workdir, _catalog = self.open_workspace({"alpha": 4})
    images = written["alpha"]
    first = self.pair_from(self.client.get("/versus/main").text)
    assert first is not None
    submit = self.client.post(
      "/versus/main/comparisons",
      data={"image_a": first[0], "image_b": first[1], "result": "A>B"},
    )
    second_page = self.client.get("/versus/main")
    second = self.pair_from(second_page.text)

    self.assertEqual(submit.status_code, 303)
    self.assertEqual(submit.headers["location"], "/versus/main")
    self.assertIsNotNone(second)
    assert second is not None
    self.assertTrue(set(first).isdisjoint(second))
    self.assertEqual(set(first) | set(second), {str(image) for image in images})
    self.assertIn('class="slot" name="result" type="submit" value="A&gt;B"', second_page.text)
    self.assertIn('class="tie" name="result" type="submit" value="A≈B">≈</button>', second_page.text)
    self.assertIn('class="slot" name="result" type="submit" value="A&lt;B"', second_page.text)
    self.assertNotIn("<figcaption>", second_page.text)
    self.assertEqual(read_rows(store), [{"image_a": first[0], "image_b": first[1], "result": "A>B"}])

  def test_undo_restores_the_last_pair(self) -> None:
    store, _written, _workdir, _catalog = self.open_workspace({"alpha": 4})
    first = self.pair_from(self.client.get("/versus/main").text)
    assert first is not None
    self.client.post("/versus/main/comparisons", data={"image_a": first[0], "image_b": first[1], "result": "A>B"})
    second = self.pair_from(self.client.get("/versus/main").text)
    assert second is not None
    self.client.post("/versus/main/comparisons", data={"image_a": second[0], "image_b": second[1], "result": "A≈B"})

    done = self.client.get("/versus/main")
    undo = self.client.post("/versus/main/comparisons/undo")
    restored_page = self.client.get("/versus/main")

    self.assertIn('data-state="done"', done.text)
    self.assertEqual(undo.status_code, 303)
    self.assertEqual(self.pair_from(restored_page.text), second)
    self.assertNotIn(first[0], restored_page.text)
    self.assertEqual(read_rows(store), [{"image_a": first[0], "image_b": first[1], "result": "A>B"}])

  def test_done_when_fewer_than_two_images_remain(self) -> None:
    _store, written, _workdir, _catalog = self.open_workspace({"alpha": 1})
    page = self.client.get("/versus/main")

    self.assertEqual(page.status_code, 200)
    self.assertIn('data-state="done"', page.text)
    self.assertIn("No pairs left.", page.text)
    self.assertNotIn('id="undo-button"', page.text)
    self.assertNotIn(str(written["alpha"][0]), page.text)

  def test_refresh_keeps_the_current_pair(self) -> None:
    self.open_workspace({"alpha": 4})
    first = self.client.get("/versus/main")
    second = self.client.get("/versus/main")
    self.assertEqual(self.pair_from(first.text), self.pair_from(second.text))
    self.assertIn('id="undo-button" type="submit" disabled', first.text)

  def test_progress_counts_track_comparisons_exclusions_and_undo(self) -> None:
    self.open_workspace({"alpha": 4})
    start = self.client.get("/versus/main")
    self.assertIn("已完成 0　未處理 4", start.text)

    first = self.pair_from(start.text)
    assert first is not None
    self.client.post("/versus/main/comparisons", data={"image_a": first[0], "image_b": first[1], "result": "A>B"})
    after_comparison = self.client.get("/versus/main")
    self.assertIn("已完成 1　未處理 2", after_comparison.text)

    second = self.pair_from(after_comparison.text)
    assert second is not None
    self.client.post("/versus/main/exclusions", data={"image": second[0]})
    after_exclude = self.client.get("/versus/main")
    self.assertIn('data-state="done"', after_exclude.text)
    self.assertIn("已完成 1　未處理 1", after_exclude.text)

    self.client.post("/versus/main/comparisons/undo")
    after_undo = self.client.get("/versus/main")
    self.assertIn("已完成 1　未處理 2", after_undo.text)

  def test_startup_arguments_create_workdir(self) -> None:
    with TemporaryDirectory() as directory:
      root = Path(directory)
      images = root / "images"
      images.mkdir()
      workdir = root / "nested" / "work"
      arguments = parse_startup_arguments(["--images-root", str(images), "--workdir", str(workdir)])
      self.assertEqual(arguments.images_root, images.resolve())
      self.assertEqual(arguments.workdir, workdir.resolve())
      self.assertTrue(workdir.is_dir())

  def test_port_defaults_to_8000(self) -> None:
    with TemporaryDirectory() as directory:
      root = Path(directory)
      images = root / "images"
      images.mkdir()
      arguments = parse_startup_arguments([
        "--images-root", str(images),
        "--workdir", str(root / "work"),
      ])
      self.assertEqual(arguments.port, 8000)

  def test_port_argument_is_accepted(self) -> None:
    with TemporaryDirectory() as directory:
      root = Path(directory)
      images = root / "images"
      images.mkdir()
      arguments = parse_startup_arguments([
        "--images-root", str(images),
        "--workdir", str(root / "work"),
        "--port", "9001",
      ])
      self.assertEqual(arguments.port, 9001)

  def test_port_rejects_values_outside_1_to_65535(self) -> None:
    with TemporaryDirectory() as directory:
      root = Path(directory)
      images = root / "images"
      images.mkdir()
      base = ["--images-root", str(images), "--workdir", str(root / "work"), "--port"]
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
      self.use_state(app.state.catalog, app.state.workdir if isinstance(app.state.workdir, Path) else root)
      with patch.object(versus_mobile.uvicorn, "run") as run:
        versus_mobile.main([
          "--images-root", str(images),
          "--workdir", str(root / "work"),
          "--port", "9001",
        ])
      run.assert_called_once_with(app, host="0.0.0.0", port=9001)

  def test_main_passes_host_to_uvicorn(self) -> None:
    with TemporaryDirectory() as directory:
      root = Path(directory)
      images = root / "images"
      images.mkdir()
      self.use_state(app.state.catalog, root)
      with patch.object(versus_mobile.uvicorn, "run") as run:
        versus_mobile.main([
          "--images-root", str(images),
          "--workdir", str(root / "work"),
          "--host", "127.0.0.1",
        ])
      run.assert_called_once_with(app, host="127.0.0.1", port=8000)

  def test_image_is_served_only_from_known_datasets(self) -> None:
    store, written, workdir, _catalog = self.open_workspace({"alpha": 2, "beta": 1}, select=["alpha"])
    available = self.client.get("/image", params={"image_path": str(written["alpha"][0])})
    other_dataset = self.client.get("/image", params={"image_path": str(written["beta"][0])})
    outside = workdir.parent / "outside.png"
    Image.new("RGB", (8, 8)).save(outside)
    missing = self.client.get("/image", params={"image_path": str(outside)})

    self.assertEqual(available.status_code, 200)
    self.assertEqual(other_dataset.status_code, 200)
    self.assertEqual(missing.status_code, 404)
    with versus_mobile.result_path(store.folder).open(newline="", encoding="utf-8") as csv_file:
      self.assertEqual(tuple(csv.DictReader(csv_file).fieldnames or ()), ("image_a", "image_b", "result"))

  def test_excluding_one_image_keeps_the_other_in_place(self) -> None:
    store, _written, _workdir, _catalog = self.open_workspace({"alpha": 4})
    first_page = self.client.get("/versus/main")
    first = self.pair_from(first_page.text)
    assert first is not None
    self.assertIn('class="exclude exclude-a" method="post"', first_page.text)
    self.assertIn(">A</button>", first_page.text)
    self.assertIn('class="exclude exclude-b" method="post"', first_page.text)
    self.assertIn(">B</button>", first_page.text)

    excluded = self.client.post("/versus/main/exclusions", data={"image": first[0]})
    second = self.pair_from(self.client.get("/versus/main").text)

    self.assertEqual(excluded.status_code, 303)
    self.assertIsNotNone(second)
    assert second is not None
    self.assertEqual(second[1], first[1])
    self.assertNotEqual(second[0], first[0])
    self.assertEqual(read_rows(store), [])
    self.assertEqual(self.excluded_images_of(store), [first[0]])

  def test_excluded_image_is_not_compared_later(self) -> None:
    store, _written, _workdir, _catalog = self.open_workspace({"alpha": 5})
    first = self.pair_from(self.client.get("/versus/main").text)
    assert first is not None
    self.client.post("/versus/main/exclusions", data={"image": first[0]})
    second = self.pair_from(self.client.get("/versus/main").text)
    assert second is not None
    self.client.post("/versus/main/comparisons", data={"image_a": second[0], "image_b": second[1], "result": "A>B"})
    third = self.pair_from(self.client.get("/versus/main").text)

    self.assertIsNotNone(third)
    assert third is not None
    self.assertNotIn(first[0], second)
    self.assertNotIn(first[0], third)
    self.assertTrue(all(first[0] not in (row["image_a"], row["image_b"]) for row in read_rows(store)))

  def test_excluding_the_last_opponent_leaves_the_other_image_unused(self) -> None:
    store, _written, _workdir, _catalog = self.open_workspace({"alpha": 2})
    first = self.pair_from(self.client.get("/versus/main").text)
    assert first is not None
    self.client.post("/versus/main/exclusions", data={"image": first[1]})
    page = self.client.get("/versus/main")

    self.assertIn('data-state="done"', page.text)
    self.assertIn("No pairs left.", page.text)
    self.assertEqual(self.excluded_images_of(store), [first[1]])

  def test_cannot_exclude_an_image_outside_the_current_pair(self) -> None:
    store, written, _workdir, _catalog = self.open_workspace({"alpha": 4})
    pair = self.pair_from(self.client.get("/versus/main").text)
    assert pair is not None
    outsider = next(str(image) for image in written["alpha"] if str(image) not in pair)
    response = self.client.post("/versus/main/exclusions", data={"image": outsider})

    self.assertEqual(response.status_code, 409)
    self.assertEqual(self.excluded_images_of(store), [])

  def test_undo_reverts_the_latest_exclusion_before_an_earlier_comparison(self) -> None:
    store, _written, _workdir, _catalog = self.open_workspace({"alpha": 4})
    first = self.pair_from(self.client.get("/versus/main").text)
    assert first is not None
    self.client.post("/versus/main/comparisons", data={"image_a": first[0], "image_b": first[1], "result": "A>B"})
    second = self.pair_from(self.client.get("/versus/main").text)
    assert second is not None
    self.client.post("/versus/main/exclusions", data={"image": second[0]})

    undo_exclusion = self.client.post("/versus/main/comparisons/undo")
    restored = self.pair_from(self.client.get("/versus/main").text)

    self.assertEqual(undo_exclusion.status_code, 303)
    self.assertEqual(restored, second)
    self.assertEqual(read_rows(store), [{"image_a": first[0], "image_b": first[1], "result": "A>B"}])
    self.assertEqual(self.excluded_images_of(store), [])

    undo_comparison = self.client.post("/versus/main/comparisons/undo")
    restored_again = self.pair_from(self.client.get("/versus/main").text)

    self.assertEqual(undo_comparison.status_code, 303)
    self.assertEqual(restored_again, first)
    self.assertEqual(read_rows(store), [])

  def test_undo_restores_an_exclusion_when_nothing_has_been_compared(self) -> None:
    store, _written, _workdir, _catalog = self.open_workspace({"alpha": 4})
    first_page = self.client.get("/versus/main")
    first = self.pair_from(first_page.text)
    assert first is not None
    self.assertIn('id="undo-button" type="submit" disabled', first_page.text)
    self.client.post("/versus/main/exclusions", data={"image": first[0]})
    excluded_page = self.client.get("/versus/main")

    self.assertIn('id="undo-button" type="submit"', excluded_page.text)
    self.assertNotIn('id="undo-button" type="submit" disabled', excluded_page.text)

    undo = self.client.post("/versus/main/comparisons/undo")
    restored = self.pair_from(self.client.get("/versus/main").text)

    self.assertEqual(undo.status_code, 303)
    self.assertEqual(restored, first)
    self.assertEqual(read_rows(store), [])
    self.assertEqual(self.excluded_images_of(store), [])

  def test_legacy_comparisons_csv_becomes_history(self) -> None:
    directory = TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    root = Path(directory.name)
    images = self.write_images(root / "images" / "alpha", 4)
    catalog = versus_mobile.build_catalog(root / "images")
    workdir = root / "work"
    folder = workdir / "legacy"
    folder.mkdir(parents=True)
    versus_mobile.write_dataset_names(folder, ("alpha",))
    with (folder / "comparisons.csv").open("w", encoding="utf-8", newline="") as csv_file:
      writer = csv.DictWriter(csv_file, fieldnames=("image_a", "image_b", "result"))
      writer.writeheader()
      writer.writerow({"image_a": str(images[0]), "image_b": str(images[1]), "result": "A>B"})
      writer.writerow({"image_a": str(images[2]), "image_b": str(images[3]), "result": "A<B"})
    store = versus_mobile.make_store(catalog, "legacy", folder)
    versus_mobile.initialize_output(store)
    self.use_state(catalog, workdir)
    self.login()

    with (folder / "history.csv").open(newline="", encoding="utf-8") as csv_file:
      self.assertEqual(list(csv.DictReader(csv_file)), [
        {"action": "comparison", "image_a": str(images[0]), "image_b": str(images[1]), "result": "A>B"},
        {"action": "comparison", "image_a": str(images[2]), "image_b": str(images[3]), "result": "A<B"},
      ])
    self.assertEqual(self.excluded_images_of(store), [])

    undo = self.client.post("/versus/legacy/comparisons/undo")
    restored = self.pair_from(self.client.get("/versus/legacy").text)

    self.assertEqual(undo.status_code, 303)
    self.assertEqual(restored, (str(images[2]), str(images[3])))
    self.assertEqual(read_rows(store), [{"image_a": str(images[0]), "image_b": str(images[1]), "result": "A>B"}])

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

  def test_refresh_catalog_includes_each_date_since_last_success(self) -> None:
    with TemporaryDirectory() as directory:
      root = Path(directory)
      images_root = root / "images"
      alpha = images_root / "alpha"
      alpha.mkdir(parents=True)
      original = alpha / "old.png"
      original.touch()
      beta = images_root / "beta"
      beta.mkdir()
      (beta / "beta.png").touch()
      catalog = versus_mobile.build_catalog(images_root)
      new_paths = []
      for day in ("2026-09-24", "2026-09-25", "2026-09-26"):
        folder = alpha / day[:7] / day
        folder.mkdir(parents=True)
        image = folder / f"{day}-1.png"
        image.touch()
        new_paths.append(str(image.resolve()))
      with patch.object(Path, "rglob", side_effect=AssertionError("no repeat full crawl")):
        refreshed = versus_mobile.refresh_catalog(catalog, date(2026, 9, 24), date(2026, 9, 26))
      self.assertEqual(refreshed.datasets["alpha"], frozenset({str(original.resolve()), *new_paths}))
      self.assertEqual(refreshed.datasets["beta"], frozenset({str((beta / "beta.png").resolve())}))
      self.assertEqual(catalog.datasets["alpha"], frozenset({str(original.resolve())}))
      self.assertIsNot(refreshed, catalog)

  def test_refresh_swaps_catalog_and_preserves_current_pair(self) -> None:
    _store, _written, _workdir, catalog = self.open_workspace({"alpha": 2})
    before = self.pair_from(self.client.get("/versus/main").text)
    day = date(2026, 9, 25)
    folder = catalog.images_root / "alpha" / "2026-09" / "2026-09-25"
    folder.mkdir(parents=True)
    new_images = [folder / f"2026-09-25-{seed}.png" for seed in (1, 2)]
    for image in new_images:
      image.touch()
    app.state.catalog = versus_mobile.refresh_catalog(catalog, day, day)
    after = self.pair_from(self.client.get("/versus/main").text)
    self.assertEqual(before, after)
    self.assertTrue({str(path.resolve()) for path in new_images} <= app.state.catalog.datasets["alpha"])

  def test_lifespan_refreshes_new_images_without_manual_trigger(self) -> None:
    with TemporaryDirectory() as directory:
      root = Path(directory)
      images = root / "images" / "alpha"
      images.mkdir(parents=True)
      catalog = versus_mobile.build_catalog(root / "images")
      workdir = root / "work"
      folder = workdir / "main"
      folder.mkdir(parents=True)
      versus_mobile.write_dataset_names(folder, ("alpha",))
      self.use_state(catalog, workdir)
      app.state.refresh_date = date(2026, 9, 25)
      with patch.object(versus_mobile, "IMAGE_REFRESH_INTERVAL_SECONDS", 0.01), \
           patch.object(versus_mobile, "hk_today", return_value=date(2026, 9, 25)):
        with TestClient(app):
          dated = images / "2026-09" / "2026-09-25"
          dated.mkdir(parents=True)
          image = dated / "2026-09-25-1.png"
          image.touch()
          deadline = time.monotonic() + 2
          while str(image.resolve()) not in app.state.catalog.datasets["alpha"] and time.monotonic() < deadline:
            time.sleep(0.01)
          self.assertIn(str(image.resolve()), app.state.catalog.datasets["alpha"])
          store = versus_mobile.make_store(app.state.catalog, "main", folder)
          self.assertIn(str(image.resolve()), versus_mobile.unused_images(store, []))

  def test_recency_weights_halve_every_128_ranks(self) -> None:
    images = [f"{index:04d}" for index in range(257)]
    weights = versus_mobile.recency_weights(images)
    self.assertEqual(weights[-1], 1)
    self.assertEqual(weights[-1 - 128], 0.5)
    self.assertEqual(weights[-1 - 256], 0.25)

  def test_recency_weights_rerank_after_removals(self) -> None:
    oldest_first = ["E", "D", "C", "B", "A"]
    original = versus_mobile.recency_weights(oldest_first)
    reranked = versus_mobile.recency_weights(["E", "C", "A"])
    self.assertEqual(reranked, [original[2], original[3], original[4]])

  def test_weighted_sample_passes_rank_weights_and_drops_the_pick(self) -> None:
    oldest_first = ["E", "D", "C", "B", "A"]
    calls: list[tuple[list[str], list[float]]] = []

    def choices(population, weights, k):
      calls.append((list(population), list(weights)))
      return [population[-1]]

    with patch.object(versus_mobile.random, "choices", side_effect=choices):
      chosen = versus_mobile.weighted_sample(oldest_first, 2)

    self.assertEqual(chosen, ["A", "B"])
    self.assertEqual(calls[0][0], oldest_first)
    self.assertEqual(calls[0][1], versus_mobile.recency_weights(oldest_first))
    self.assertEqual(calls[1][0], ["E", "D", "C", "B"])
    self.assertEqual(calls[1][1], versus_mobile.recency_weights(["E", "D", "C", "B"]))
    self.assertNotIn("A", calls[1][0])

  def test_list_opens_compare_directly(self) -> None:
    self.open_workspace({"alpha": 2})
    listing = self.client.get("/")
    self.assertIn('class="versus-name" href="/versus/main"', listing.text)
    self.assertIn('href="/versus/main/settings"', listing.text)
    self.assertIn('href="/new"', listing.text)
    self.assertNotIn('action="/new"', listing.text)
    self.assertNotIn('name="name"', listing.text)
    self.assertIn("alpha", listing.text)
    compare = self.client.get("/versus/main")
    self.assertIn('data-state="compare"', compare.text)
    self.assertIn('class="back" href="/"', compare.text)

  def test_create_requires_a_dataset_and_rejects_bad_names(self) -> None:
    self.open_workspace({"alpha": 1}, versus=None)
    form = self.client.get("/new")
    missing = self.client.post("/new", data={"name": "demo"})
    invalid = self.client.post("/new", data={"name": "../nope", "datasets": "alpha"})
    unknown = self.client.post("/new", data={"name": "demo", "datasets": "missing"})
    created = self.client.post("/new", data={"name": "demo", "datasets": "alpha"})
    duplicate = self.client.post("/new", data={"name": "demo", "datasets": "alpha"})
    page = self.client.get("/new?error=exists")

    self.assertIn('action="/new"', form.text)
    self.assertIn('href="/"', form.text)
    self.assertEqual(missing.headers["location"], "/new?error=datasets")
    self.assertEqual(invalid.headers["location"], "/new?error=name")
    self.assertEqual(unknown.headers["location"], "/new?error=unknown")
    self.assertEqual(created.headers["location"], "/versus/demo")
    self.assertEqual(duplicate.headers["location"], "/new?error=exists")
    self.assertIn("名稱已存在。", page.text)
    self.assertNotIn('class="versus-name"', page.text)

  def test_delete_removes_the_versus_folder(self) -> None:
    store, _written, _workdir, _catalog = self.open_workspace({"alpha": 2})
    confirm = self.client.get("/versus/main/delete")
    deleted = self.client.post("/versus/main/delete")
    listing = self.client.get("/")

    self.assertIn("刪除 main", confirm.text)
    self.assertEqual(deleted.status_code, 303)
    self.assertEqual(deleted.headers["location"], "/")
    self.assertFalse(store.folder.exists())
    self.assertIn("沒有 versus。", listing.text)
    self.assertNotIn('href="/versus/main"', listing.text)

  def test_mixed_pool_pairs_images_from_every_selected_dataset(self) -> None:
    _store, written, _workdir, _catalog = self.open_workspace({"alpha": 1, "beta": 1})
    pair = self.pair_from(self.client.get("/versus/main").text)
    self.assertEqual(pair and set(pair), {str(written["alpha"][0]), str(written["beta"][0])})

  def test_changing_datasets_keeps_history_and_updates_the_pool(self) -> None:
    store, written, _workdir, catalog = self.open_workspace({"alpha": 2, "beta": 2})
    start = self.client.get("/versus/main")
    self.assertIn("未處理 4", start.text)
    pair = self.pair_from(start.text)
    assert pair is not None
    self.client.post("/versus/main/comparisons", data={"image_a": pair[0], "image_b": pair[1], "result": "A>B"})

    update = self.client.post("/versus/main/settings", data={"datasets": "alpha"})
    settings = self.client.get("/versus/main/settings")
    refreshed = versus_mobile.make_store(catalog, "main", store.folder)
    history = versus_mobile.read_history(refreshed)
    unused = versus_mobile.unused_images(refreshed, history)
    alpha_paths = {str(image) for image in written["alpha"]}
    page = self.client.get("/versus/main")

    self.assertEqual(update.headers["location"], "/versus/main/settings")
    self.assertIn('value="alpha" checked', settings.text)
    self.assertNotIn('value="beta" checked', settings.text)
    self.assertEqual(versus_mobile.comparison_rows(history), [{"image_a": pair[0], "image_b": pair[1], "result": "A>B"}])
    self.assertEqual(unused, alpha_paths - {pair[0], pair[1]})
    self.assertIn(f"已完成 1　未處理 {len(unused)}", page.text)

  def test_pairs_stay_with_their_versus(self) -> None:
    _store, written, _workdir, _catalog = self.open_workspace({"alpha": 2, "beta": 2}, versus=None)
    self.client.post("/new", data={"name": "main", "datasets": "alpha"})
    self.client.post("/new", data={"name": "other", "datasets": "beta"})
    main_pair = self.pair_from(self.client.get("/versus/main").text)
    other_pair = self.pair_from(self.client.get("/versus/other").text)
    again = self.pair_from(self.client.get("/versus/main").text)

    self.assertEqual(main_pair and set(main_pair), {str(image) for image in written["alpha"]})
    self.assertEqual(other_pair and set(other_pair), {str(image) for image in written["beta"]})
    self.assertEqual(again, main_pair)

  def test_undo_samples_the_current_pool_when_restored_images_are_not_selected(self) -> None:
    store, written, _workdir, catalog = self.open_workspace({"alpha": 2, "gamma": 2}, select=["alpha"])
    pair = self.pair_from(self.client.get("/versus/main").text)
    assert pair is not None
    self.client.post("/versus/main/comparisons", data={"image_a": pair[0], "image_b": pair[1], "result": "A>B"})
    self.client.post("/versus/main/settings", data={"datasets": "gamma"})
    undo = self.client.post("/versus/main/comparisons/undo")
    restored = self.pair_from(self.client.get("/versus/main").text)
    refreshed = versus_mobile.make_store(catalog, "main", store.folder)

    self.assertEqual(undo.status_code, 303)
    self.assertEqual(restored and set(restored), {str(image) for image in written["gamma"]})
    self.assertEqual(read_rows(refreshed), [])

  def test_missing_versus_is_not_found(self) -> None:
    self.open_workspace({"alpha": 1}, versus=None)
    response = self.client.get("/versus/missing")
    self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
  unittest.main()
