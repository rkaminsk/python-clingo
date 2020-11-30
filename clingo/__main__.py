from clingo import Control

def on_model(m):
    print(m)

ctl = Control(['0'])
ctl.add("base", [], "1 {a; b} 1. c.")
ctl.ground([("base", [])])
ctl.solve(on_model=on_model)
