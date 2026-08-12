from scripts.dependency_inventory import build_inventory


def test_dependency_inventory_is_locked_and_self_bound():
    inventory = build_inventory()
    assert inventory["schema_version"] == 1
    assert inventory["packages"]
    assert len(inventory["inventory_sha256"]) == 64
    assert all({"name", "version"} <= set(package) for package in inventory["packages"])
