import generateXML as gxml
from generateXML import XMLTree


t = XMLTree()
t.add_compiler()
t.add_default()
t.add_option()

t.generate()