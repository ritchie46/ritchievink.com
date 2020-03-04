# first compress and resize images
# only images wider than 800px are resized.
echo compress jpegs
mogrify -resize '800>' -format jpg -quality 85 static/img/**/*.{jpg,png}

echo compress pngs
# png compression is lossless so we go for maximum compression
for f in static/img/**/*.png; do 
	convert -resize '800>' -format png -quality 1 $f $f
done

echo build site and rsync
hugo && rsync --exclude='*.swp' -v -r public/. linode:/var/www/blog

echo delete generated images
# remove the generated jpg files and only keep originals.
git clean -f static/img/
git checkout -- static/img/

