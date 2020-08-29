#!/bin/sh

# create a .zip of the module so it can be deployed to Hadoop
rm -f deploy/project.zip
zip -q -r deploy/project.zip project/
