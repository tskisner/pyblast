## Package Data Files

Any files you place here will be installed along with the package,
and will be available by using pkg_resources.  For example, to get
the path to this README you would do:

    from pkg_resources import resource_filename
    path = resource_filename(
        "pyblast.tests", os.path.join("data", "README.md")
    )
