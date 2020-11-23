---
id: extraction
sidebar_label: 3. Extraction
title: Extraction Module
slug: /extraction-module
---

import CodeBlock from "../src/utils/CodeBlock"
import OutputBlock from "../src/utils/OutputBlock"

We started from the detection of student-id(s) followed by the correction of their orientations and now we have finally reached the extraction module. Once more we shall recall that this final module which puts an end to our Information Extraction pipeline aims at extracting information from fields within student-id(s). In this sense, we shall now ask ourselves how to individually extract the fields within student-id(s) which are of interest to us ? The answer to that question is quite simple and is nothing Pytorch related.

For an individual field, we shall use the coordinates of the size of a student id. The image below better illustrates our theory.
![img](../static/img/field-coordinates.svg)

From the image above, the ratio of the size of a field relative to the size of a fixed-sized document remains constant no matter the size of the document images.

The relative coordinates were extrapolated prior to this module and stored in the ``./code/field_coordiinates.yaml`` file.

Consequently,  shall use these coordinates to crop every field containing meaningful information from aligned documents to effectuate information extraction.

Let's load the fields' coordinates.
<CodeBlock file="extraction_coordinates_init"></CodeBlock>
<OutputBlock file="extraction_coordinates_init_output"></OutputBlock>

Finally, let's use the fields' coordinates to extract the information fields.
<CodeBlock file="extraction_predictions"></CodeBlock>

![img](../static/img/extraction-prediction.svg)
