    def set_terrain(self,
                    type="default",
                    name: Union[str, None] = None,
                    size: List[str] = None):
        '''
        定义地形，为worldbody的子元素
        '''
        tag = "geom"
        if type == "default":
            if not name:
                name = "flatfloor"
            if not size:
                Size = "20 20 0.125"
            else:
                Size = self.list2str(size)
            terrain_attr = {"condim": "3",
                            "friction": "1 .1 .1",
                            "material": "MatPlane",
                            "name": name,
                            "pos": "0 0 0",
                            "rgba": "0.8 0.9 0.8 1",
                            "size": Size,
                            "type": "plane", }
            self.elements["worldbody"].child_element(tag, terrain_attr)

        '''
        定义楼梯/梯子
        '''
        # Define the parameters for the box geometries
        box_size = ".03 1.2 .01"
        box_rgba = "0 .9 0 1"
        box_condim = "3"
        box_friction = "1 .1 .1"

        # Define the positions using an arithmetic sequence with a common difference of 0.2
        positions = [(round(i * 0.2, 3), 0, round(i * 0.2 * 2 / 3, 3)) for i in range(11)]

        # Create a body element for the boxes,named "ground"

        ground_attr = {"name": "ground"}
        ground = self.elements["worldbody"].child_element("body", ground_attr)
        # Create a geometry for each position
        for i, pos in enumerate(positions):
            box_attr = {
                "type": "box",
                "size": box_size,
                "pos": f"{pos[0]} {pos[1]} {pos[2]}",
                "rgba": box_rgba,
                "condim": box_condim,
                "friction": box_friction,
            }
            ground.child_element(tag, box_attr)








        '''
        定义地形end
        '''
