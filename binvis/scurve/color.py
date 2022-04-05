import string


class ColorClass:
    """Convert binary value to RBG values"""
    def __init__(self, data):
        self.data = data  # The binary values
        s = list(set(data))
        s.sort()
        self.symbol_map = {v: i for (i, v) in enumerate(s)}

    def point(self, x):
        """Get The RGB Values

        Args:
            x (int): Index of the binary value in the list

        Returns:
            list: The RGB values
        """
        c = self.data[x]
        
        if c == 0:
            return [0, 0, 0]
        elif c == 255:
            return [255, 255, 255]
        elif (0 < c < 32) and ((c != 9) or (c != 10) or (c != 13)):
            return [104, 172, 87]
        elif chr(c) in string.printable:
            return [55, 126, 184]
        return [228, 26, 28]
    
    
    def __len__(self):
        return len(self.data)
