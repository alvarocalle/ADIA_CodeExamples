#!/bin/bash
echo "Installing Libraries"
while IFS=" " read -r package version; 
do 
  echo $package
  emptyVar=""
  if [ "$version" = "$emptyVar" ]
  then
    Rscript -e "devtools::install_version('"$package"')"; 
  else
    Rscript -e "devtools::install_version('"$package"', version='"$version"')"; 
  fi
done < "/app/requirements.txt"
echo "Done installing Libraries"