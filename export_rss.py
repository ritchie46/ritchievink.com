#!/usr/bin/python3
import xml.etree.ElementTree as ET

tree = ET.parse("public/index.xml")
channel = tree.getroot()[0]

items = [a for a in channel if a.tag == "item"]

html_link = "<a href='{}'>{}</a>"
s = "<ul>\n"

for i in range(5):
    link = html_link.format(items[i].find("link").text, items[i].find("title").text)
    s += "\t<li>{}</li>\n".format(link)
s += "</ul>"

print(s)
