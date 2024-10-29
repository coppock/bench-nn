import shlex

class Shape:
    def __init__(self, s):
        parser = shlex.shlex(s, posix=True)
        parser.whitespace = ':'
        parser.whitespace_split = True
        self.name, shapes = list(parser)

        shapes = [[int(i) for i in shape.split('x')]
                  for shape in shapes.split(',')]
        try: self.min_shape, self.opt_shape, self.max_shape = shapes
        except ValueError:
            self.opt_shape, = shapes
            self.min_shape, self.max_shape = 2*[self.opt_shape]
