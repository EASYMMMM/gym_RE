import xml.etree.ElementTree as ET

def extract_geom_positions(xml_file):
    # load xml file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # find the ground body
    ground_body = root.find("./worldbody/body[@name='ground']")

    positions = []
    # iterate over each geom in the ground body
    for geom in ground_body.iter('geom'):
        # extract xyz coordinates of the geom's position attribute
        x, y, z = map(float, geom.attrib['pos'].split())
        positions.append((x, y, z))

    return positions



positions = extract_geom_positions('e.xml')
print(positions)
