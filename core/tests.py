import unittest


from system import Component, System
from scheduler import Activity
from utils import components, structure


class ComponentTestCase(unittest.TestCase):

    def test_components_declaration(self):
        for c in components:
            self.assertIsInstance(c, Component)


class SystemTestCase(unittest.TestCase):

    def test_system_declaration(self):
        system = System(
            structure=structure,
            resources=3,
            components=components,
        )
        self.assertIsInstance(system, System)


class ActivityTestCase(unittest.TestCase):

    def setUp(self):
        return super().setUp()

if __name__ == "__main__":
    unittest.main()
