## Usage ##
This is a simple tool to parse a directory of `.json` files produced as the output of
a CVAT annotation session, saved in a COCO 1.0 format, and consolidate
the annotations and image IDs into a `.csv` file. This form is then passed onto
whichever tool will be used to pre-process the data further or train an ML model.

This tool is written in OCaml. To run the script, follow the following directions:

1) First install OCaml. The 
[preface of the CS 3110 textbook](https://cs3110.github.io/textbook/chapters/preface/install.html) 
is the best resource for this.

2) Install some dependencies. Run `opam install csv` and `opam install yojson`.
Then, run `opam update` to update the `opam` packages database. Finally, run
`opam upgrade` to bring all the packages to their latest versions.

3) Compile the project with `dune build`. Then, from the directory `utils/cvat_data_annotation`, run
`dune exec bin/main.exe <path to directory of .jsons> <output path of generated .csv> <competition type (e.g. cuasc-25)>`. 

## Things Breaking? ##
This script doesn't really have a lot of crazy dependencies. If something is breaking,
it possible that your `opam` package versions are too far behind. Running
`opam {update, upgrade}` should fix these.
