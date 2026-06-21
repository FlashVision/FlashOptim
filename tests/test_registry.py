"""Tests for the Registry class."""

import pytest

from flashoptim.registry import Registry, BACKBONES, PRUNERS, QUANTIZERS


class TestRegistry:
    """Tests for the Registry pattern."""

    def test_create_registry(self):
        reg = Registry("test_registry")
        assert reg.name == "test_registry"
        assert len(reg) == 0

    def test_register_class(self):
        reg = Registry("test")

        @reg.register("MyClass")
        class MyClass:
            pass

        assert "MyClass" in reg
        assert len(reg) == 1

    def test_register_auto_name(self):
        reg = Registry("test")

        @reg.register()
        class AutoNamed:
            pass

        assert "AutoNamed" in reg

    def test_build_instance(self):
        reg = Registry("test")

        @reg.register("Foo")
        class Foo:
            def __init__(self, value=42):
                self.value = value

        instance = reg.build("Foo", value=99)
        assert instance.value == 99

    def test_build_unknown_raises(self):
        reg = Registry("test")
        with pytest.raises(KeyError, match="not found"):
            reg.build("NonExistent")

    def test_duplicate_register_raises(self):
        reg = Registry("test")

        @reg.register("Dup")
        class Dup1:
            pass

        with pytest.raises(KeyError, match="already registered"):

            @reg.register("Dup")
            class Dup2:
                pass

    def test_list(self):
        reg = Registry("test")

        @reg.register("B")
        class B:
            pass

        @reg.register("A")
        class A:
            pass

        assert reg.list() == ["A", "B"]

    def test_get(self):
        reg = Registry("test")

        @reg.register("Item")
        class Item:
            pass

        assert reg.get("Item") is Item
        assert reg.get("Missing") is None

    def test_contains(self):
        reg = Registry("test")

        @reg.register("Present")
        class Present:
            pass

        assert "Present" in reg
        assert "Absent" not in reg

    def test_repr(self):
        reg = Registry("my_reg")
        r = repr(reg)
        assert "Registry" in r
        assert "my_reg" in r

    def test_global_registries_exist(self):
        assert BACKBONES.name == "backbones"
        assert PRUNERS.name == "pruners"
        assert QUANTIZERS.name == "quantizers"
