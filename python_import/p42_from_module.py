from machine.car import drive
from machine.tv import watch

drive()
watch()

print('=================================')

from machine import car
from machine import tv
# from machine import car, tv   <- 같은 폴더에 있으면 이렇게도 됨
car.drive()
tv.watch()