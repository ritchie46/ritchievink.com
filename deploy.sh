# first compress and resize images (resizing is only upper bound per dimension)
mogrify -resize 800x600 -format jpg -quality 70 static/img/**/*.{jpg,png}

hugo && rsync -v -r public/. linode:/var/www/blog

# remove the generated jpg files and only keep originals.
git clean -f static/img/
git checkout -- static/img/

