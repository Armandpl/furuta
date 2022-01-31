# Hardware

## Bill of Materials
We tried selecting components available worldwide. Feel to open an issue if you have trouble sourcing one or if you know of a better alternative. 

| Item Name (+datasheet link)                                                                                                                                                                            | Supplier                                                                                                                                                              | Price (EUR) |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| [Nvidia Jetson Nano 2GB Developer Kit](https://developer.nvidia.com/embedded/learn/jetson-nano-2gb-devkit-user-guide)                                                                                  | [Amazon.fr](https://www.amazon.fr/Waveshare-Jetson-Developer-Hands-Robotics/dp/B08M5J1WM2)                                                                            | 87.99       |
| [CUI Devices AMT103-V (Pendulum Encoder)](https://www.cuidevices.com/product/resource/amt10.pdf)                                                                                                       | Digikey (#102-1308-ND)                                                                                                                                                | 21.04       |
| CUI Devices wide base                                                                                                                                                                                  | Digikey (#102-3323-ND)                                                                                                                                                | 1.36        |
| [2x MIKROE LS7366R Counter Click (Encoder Counters)](https://lsicsi.com/datasheets/LS7366R.pdf)                                                                                                        | Digikey (#1471-1495-ND)                                                                                                                                               | 2x 22.05    |
| [Sparkfun Slip Ring 6 wires 2A](https://cdn.sparkfun.com/datasheets/Robotics/SNM022A-06%20update.pdf)                                                                                                  | Digikey (#1568-1253-ND)                                                                                                                                               | 13.18       |
| 2x 8x22x7mm Ball Bearing                                                                                                                                                                               | Digikey (#1995-1010-ND)                                                                                                                                               | 2x 0.88     |
| [MC33926 Motor Driver Carrier](https://www.pololu.com/product/1212)                                                                                                                                    | [Polulu](https://www.pololu.com/product/1212) or [generationrobots.com](https://www.generationrobots.com/fr/400946-carte-de-pilotage-mc33926-pour-deux-moteurs-.html) | 22.33       |
| [Polulu 25D Gearmotor (4.4:1) Medium Power 12V with Encoder](https://www.pololu.com/product/4861)                                                                                                      | [Polulu](https://www.pololu.com/product/4861)                                                                                                                         | 33.5        |
| 12V Power Supply                                                                                                                                                                                       |                                                                                                                                                                       | <50         |
| 8mm x 1m Aluminium Tube                                                                                                                                                                                | local hardware shop                                                                                                                                                   | 4.18        |
| Neewer Boom Arm                                                                                                                                                                                        | [Amazon.fr](https://www.amazon.fr/Neewer-Support-Microphone-Enregistrement-Broadcasting/dp/B00DY1F2CS/)                                                               | 19.11       |
| Nuts and bolts: <br/> <ul> <li>2x M10 Nuts</li> <li>6x M2 x 4mm x 3.5mm Threaded Inserts</li> <li>6x M2 x 7mm Screws</li> <li>4x M3 x 4mm x 5mm Threaded Inserts</li> <li>6x M3 x 6mm Screws</li> </li> <li>2x M3 Nuts</li></ul> | Amazon/local hardware shop                                                                                                                  | <20         |
| 3x Zip-Ties                                                                                                                                                                                            | Amazon/local hardware shop                                                                                                                                            | 1           |
| 150x150x20mm Wood Scrap                                                                                                                                                                                |                                                                                                                                                                       | 0           |
| Total                                                                                                                                                                                                  |                                                                                                                                                                       | 319.55      |

### Note
- I had to cut down some of the M3x6mm screws listed above for them to fit with the printed shaft colars. Also, when tightening too hard the threaded inserts can come off. If you can source 8mm shaft collars you are better off doing that.
- I got my slip ring second hand with the label scratched off. The one listed above is the closest thing I found on Digikey. You need a slip ring for data transmission (low amperage) with at least 5 wires.
- I use a lab power supply I got off ebay for about 50EUR. You can definitely get away with something cheaper as long as it matches your motor stall current (1.8A for the one listed above).

## Required Tools for the Build
- 3D Printer (or access to one)
- A soldering iron to set the threaded inserts and solder connectors.
- A Dupont Crimping Kit (AWG28-18) to make nice wires and connectors. I got mine [from Amazon](https://www.amazon.fr/Kamtop-Sertissage-Sertisseuse-Connecteurs-0-1-1-0mm²/dp/B078K9DT69) for <30EUR.
- A decent wire stripper also helps. I like [this one](https://www.amazon.fr/Jokari-T20050-Pince-dénuder-automatique/dp/B002BDNL4Q/) very much.

## Assembly Instructions
These instructions are quite brief, if you're missing anything DM me [@armand_dpl](https://twitter.com/armand_dpl).

### 3D Print the Mechanical Assembly
- Print all of the STLs under `/CAD/stl`
  - Print the motor_mount.stl at 100% infill
  - Print the rest at 35% infill with 2 walls.
  - Print shaft_collar.stl twice 
  - Pause the weights.stl print at z=10mm, insert the M10 nut and resume the print. The procedure will depend on your printer and slicer software.
- Insert the threaded inserts in the weight, arm and shaft collars prints.

### Assemble the Robot
- Secure the motor to its motor mount using two screws and one zip-tie.
- Screw the motor mount into the wood scrap (or any heavy object you wish to use as the robot base).
- Press the two bearing into the arm print
- Slide the arm onto the motor shaft. The shaft should be D-shaped and the arm print should have a D-shaped hole. Match those then secure the arm by tightening one screw through the threaded insert.
- Bend your 8mm aluminium shaft, pass it throught the bearings and secure it using the shaft collars. The aluminium shaft might be slightly to big for the bearings, if so sand it lightly until it fits.
- Attach the pendulum encoder to the shaft. Follow the instructions from the documentation to do so.
- Secure the weight to the end of the shaft.

### Connect the Electronics
- Connect both counter chips to the SPI bus on the Jetson Nano. I used one bus for each counter chip but using only one should work. **Note which bus they are connected to and which Chip Select pins you used.**
- Connect the motor encoder to one counter chip.
- Connect the pendulum encoder to the other counter chip through the slip ring.
- _WIP: add motor driver instructions + double check everything, burned one rpi by inverting vcc and vin which are literally next to each other_
- write the config file _WIP explain how to write config file + check everything in the gym is derived from the config_

