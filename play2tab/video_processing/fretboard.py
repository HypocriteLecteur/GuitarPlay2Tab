class Fretboard:
    def __init__(self, frets, strings, oriented_bb) -> None:
        self.frets = frets
        self.strings = strings
        self.oriented_bb = oriented_bb
    
    def resize(self, factor):
        self.frets[:, 0] = self.frets[:, 0] * factor
        self.strings[:, 0] = self.strings[:, 0] * factor
        self.oriented_bb = ((self.oriented_bb[0][0] * factor, self.oriented_bb[0][1] * factor), 
                            (self.oriented_bb[1][0] * factor, self.oriented_bb[1][1] * factor),
                            self.oriented_bb[2])
        return self
    
    def copy(self):
        return Fretboard(self.frets.copy(), self.strings.copy(), (self.oriented_bb[0], self.oriented_bb[1], self.oriented_bb[2]))