# Covid simulation based on a monomer model

![Simulation illustration](sim.png?raw=true)

## Monomer model

Monomers are in a box of a fixed height and width. They cannot go outside the box : when they hit a wall they bounce on hit with an elastic model of collision. It's the same for interparticle collisions.

## Covid simulation

To transform this model in a covid simulation, each monomer has an health state : healthy, sick or recovered. When a sick particule touch a healthy particle it becomes sick too for an amount of time following a normal distribution. After that it takes the recovered state. A recovered particles cannot become sick again.

## Free-for-all

When everybody can move freely, the disease spreads quickly.

![free-for-all](free.gif?raw=true)

## Lockdown

A lockdown is modeled with monomers which speed are null that represent people staying at home. However there are still people moving.

![Lockdown](lockdown.gif?raw=true)

## Closing border

A closed border is modeled by a wall in the simulation.

![Closing border short](frontier_short.gif?raw=true)

![Closing border long](frontier_long.gif?raw=true)

## Run the program

Just run `python event_code_classMonomer.py`.
You can adjust the parameters of the simulation in the file  `event_code_classMonomer.py`.

## About

A project made by Isabelle Andre et Louis Grandvaux from the ESPCI.
