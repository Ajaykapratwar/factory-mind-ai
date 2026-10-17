"""
Tests that all 10 pages in the Dash registry return HTTP 200 OK
and that the layouts can be rendered without errors.
"""

from factorymind_pharma.dashboard.app import app
import dash


def test_all_pages_http_200():
    client = app.server.test_client()
    assert len(dash.page_registry) == 10, f"Expected 10 pages, found {len(dash.page_registry)}"

    for page_key, page_info in dash.page_registry.items():
        url = page_info["path"]
        resp = client.get(url)
        assert resp.status_code == 200, f"Page {page_info['name']} at {url} returned {resp.status_code}"


def test_page_layouts_callable():
    for page_key, page_info in dash.page_registry.items():
        layout = page_info.get("layout")
        if callable(layout):
            res = layout()
            assert res is not None, f"Layout function for {page_info['name']} returned None"
        else:
            assert layout is not None, f"Layout for {page_info['name']} is None"
