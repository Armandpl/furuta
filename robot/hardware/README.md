# Hardware

## Bill of Materials
We tried selecting components available worldwide. Feel to open an issue if you have trouble sourcing one or if you know of a better alternative.

digikey cart link: https://www.digikey.com/short/4n554dq5
|                                                                                                                                                                                                                                                                                                                                                                   Item Name (+datasheet link) | Supplier         | Price (USD)            |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|------------------|------------------------|
| 1x PCB                                                                                                                                                                                                                                                                                                                                                                                        | JLPCB, PCBWAY... | 2 (+20 intl. shipping) |
| 1x 9.7:1 Metal Gearmotor 25Dx63L mm MP 12V with 48 CPR Encoder                                                                                                                                                                                                                                                                                                                                | Pololu           | 45.95                  |
| 1x TI DRV8871                                                                                                                                                                                                                                                                                                                                                                                 | Digikey          | 3                      |
| 1x TI S0108E                                                                                                                                                                                                                                                                                                                                                                                  | Digikey          | 1.6                    |
| XIAO SAMD 21                                                                                                                                                                                                                                                                                                                                                                                  | Digikey          | 5.4                    |
| 3x 0.1uf 50V (0805) capacitors                                                                                                                                                                                                                                                                                                                                                                | Digikey          | 1                      |
| 1x 18k (0805) resistor                                                                                                                                                                                                                                                                                                                                                                        | Digikey          | 1                      |
| 1x 22uf 63V aluminium cap                                                                                                                                                                                                                                                                                                                                                                     | Digikey          | 1.77                   |
| 1x CUI Devices AMT103-V                                                                                                                                                                                                                                                                                                                                                                       | Digikey          | 23.86                  |
| 1x CUI Devices Wide Base                                                                                                                                                                                                                                                                                                                                                                      | Digikey          | 1.54                   |
| 2x 8x22x7mm ball bearings                                                                                                                                                                                                                                                                                                                                                                     | Digikey          | 2                      |
| 1x adafruit slip ring                                                                                                                                                                                                                                                                                                                                                                         | Digikey          | 14.95                  |
| 12V 2A PSU                                                                                                                                                                                                                                                                                                                                                                                    | Amazon           | 10                     |
| 2x 8mm rods                                                                                                                                                                                                                                                                                                                                                                                   | Amazon           | 10                     |
| fasteners:<br>- 2x M3x8, countersunk head to secure the motor to the mount<br>- 4x M3x20mm, countersunk + 4x M3 nuts to secure the base to the clamp<br>- 1x M12x60 for the clamp<br>- 2xM3x8mm screws with countersunk head +2xM3 hex nuts (to attach the encoder to the arm)<br>- 3xM3x5mm set screws + 3xM3 square nuts (to make shaft collars)<br>- 4x M2x8mm screws (to secure the PCB) |                  |                        |

## Required Tools for the Build
- 3D Printer (or access to one)
- A soldering iron to set the threaded inserts and solder connectors.
- A Dupont Crimping Kit (AWG28-18) to make nice wires and connectors. I got mine [from Amazon](https://www.amazon.fr/Kamtop-Sertissage-Sertisseuse-Connecteurs-0-1-1-0mm²/dp/B078K9DT69) for <30EUR.
- A decent wire stripper also helps. I like [this one](https://www.amazon.fr/Jokari-T20050-Pince-dénuder-automatique/dp/B002BDNL4Q/) very much.

## Assembly Instructions
These instructions are quite brief, if you're missing anything DM me [@armand_dpl](https://twitter.com/armand_dpl).

### 3D Print the Mechanical Assembly
- Print all of the STLs under `/CAD/stl`
  - Print `motor_mount.stl` at 100% infill
  - Print the rest at 35% infill with 2 walls.
  - Print `shaft_collar.stl` twice
  - Pause the `weights.stl` print at `z=10mm`, insert the M10 nut and resume the print. The procedure will depend on your printer and slicer software.
- Insert the threaded inserts in the weight, arm and shaft collars prints.

### Assemble the Robot
- Secure the motor to its motor mount using two screws and one zip-tie.
- Screw the motor mount into the wood scrap (or any heavy object you wish to use as the robot base).
- Press the two bearing into the arm print
- Slide the arm onto the motor shaft. The shaft should be D-shaped and the arm print should have a D-shaped hole. Match those then secure the arm by tightening one screw through the threaded insert.
- Bend your 8mm aluminium shaft, pass it throught the bearings and secure it using the shaft collars. The aluminium shaft might be slightly too big for the bearings, if so sand it lightly until it fits.
- Attach the pendulum encoder to the shaft. Follow the instructions from the documentation to do so.
- Secure the weight to the end of the shaft.

### Connect the Electronics
- Connect both counter chips to the SPI bus on the Jetson Nano. I used one bus for each counter chip but using only one should work. **Note which bus they are connected to and which Chip Select pins you used.**
- Connect the motor encoder to one counter chip.
- Connect the pendulum encoder to the other counter chip through the slip ring.
- _WIP: add motor driver instructions + double check everything, burned one rpi by inverting vcc and vin which are literally next to each other_
- write the config file _WIP explain how to write config file + check everything in the gym is derived from the config_

# WIP Notes
- if issues uploading to the xiao samd/opening the serial port it might be a right issue `sudo chmod a+rw /dev/ttyACM0`
