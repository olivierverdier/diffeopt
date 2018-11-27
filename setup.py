from distutils.core import setup

setup(name='diffeopt',
      version='0.1',
      packages=['diffeopt',
                'diffeopt.group',
                'diffeopt.group.ddmatch',
                'diffeopt.group.ddmatch.action',
                'diffeopt.cometric',
                'diffeopt.distance',
      ],
      )
