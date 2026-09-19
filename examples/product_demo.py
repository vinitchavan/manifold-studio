"""No neural model required. Run after installing this repository."""
from manifold_studio.cli import main

if __name__ == "__main__":
    main(["demo", "--geometry", "sphere*torus*plane", "--out", "outputs/product_demo"])
