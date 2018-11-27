class Deformation:
    """
    A container for a big or small deformation.
    """
    def __init__(self, group):
        self.group = group
        self.reset()

    def reset(self):
        self.velocity = self.group.zero()

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        self._velocity = velocity
        self.deformation = self.group.exponential(velocity)
