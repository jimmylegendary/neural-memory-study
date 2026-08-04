function allShapes(slide) {
  if (Array.isArray(slide.shapes?.items)) return slide.shapes.items;
  if (Number.isInteger(slide.shapes?.count) && typeof slide.shapes.getItem === "function") {
    return Array.from({ length: slide.shapes.count }, (_, index) => slide.shapes.getItem(index));
  }
  return [];
}

/**
 * Imported PPTX line/connector elements can encode direction with negative cx/cy.
 * artifact-tool renders those frames but rejects them when exporting back to PPTX.
 * Convert the signed frame into a non-negative bounding box plus flip flags, which
 * is the equivalent OOXML representation and preserves the line's direction.
 */
export function normalizeImportedNegativeExtents(presentation) {
  const changes = [];
  for (const [slideIndex, slide] of presentation.slides.items.entries()) {
    for (const shape of allShapes(slide)) {
      const frame = shape.frame;
      if (!frame || (frame.width >= 0 && frame.height >= 0)) continue;
      const position = shape.position.toJSON();
      const horizontalFlip = Boolean(position.horizontalFlip) !== (frame.width < 0);
      const verticalFlip = Boolean(position.verticalFlip) !== (frame.height < 0);
      const normalized = {
        left: frame.width < 0 ? frame.left + frame.width : frame.left,
        top: frame.height < 0 ? frame.top + frame.height : frame.top,
        width: Math.abs(frame.width),
        height: Math.abs(frame.height),
        rotation: position.rotation,
        horizontalFlip,
        verticalFlip,
      };
      shape.position.set(normalized);
      changes.push({
        slide: slideIndex + 1,
        shapeId: String(shape.id),
        name: shape.name || "",
        before: frame,
        after: shape.frame,
      });
    }
  }
  return changes;
}
