import machine.car
import machine.tv

# car.drive() <- 안됨
# drive() <- 안됨
machine.car.drive()
machine.tv.watch()

########################################################

import machine.car as car
import machine.tv as tv

car.drive()
tv.watch()