
## RAW DATA

```json
{
"images" : [image],
"annotations" : [annotation],
}
image{
"image_id" : int,
"url": [string]
}
annotation{
"image_id" : int,
"label_id" : int
}
```

## Preprocessed

```json
{
    id: {'image': path, 'label_id': label_id, "flipped": False}
}
```

## ResNet

1. Center image: 2 * img/255 - 1.
