import React from 'react';

const MARKER_COLOR = '#AA4499';

const predictionStyle = { color: MARKER_COLOR, fontWeight: 'bold' };

const BracketCell = ({ char, marginRight }) => (
    <span style={{ display: 'inline-block', textAlign: 'center', marginRight }}>
        <span style={{ display: 'block', fontSize: '0.75em' }}>&nbsp;</span>
        <span style={{ display: 'block', ...predictionStyle }}>{char}</span>
    </span>
);

const ScansionDisplay = ({ scansion }) => {
    if (!scansion || scansion.length === 0) {
        return null;
    }

    return (
        <div className="mt-4">
            <h6 className="mb-1 fw-bold">Scansion</h6>
            <small className="text-muted fst-italic d-block mb-2">
                Restored text scansion.
            </small>
            <div
                className="border rounded p-3"
                style={{ overflowX: 'auto', backgroundColor: '#fdfdfd' }}
            >
                {scansion.map((entry, lineIndex) => {
                    const hasSegments = Array.isArray(entry.segments) && entry.segments.length === entry.syllables.length;
                    // for cross-syllable bracket boundary detection
                    const flatSegments = hasSegments ? entry.segments.flat() : [];
                    const flatOffsets = [];
                    if (hasSegments) {
                        let offset = 0;
                        entry.segments.forEach((syllableSegments) => {
                            flatOffsets.push(offset);
                            offset += syllableSegments.length;
                        });
                    }
                    const predictedSet = new Set(entry.prediction_syllables || []);

                    return (
                        <div
                            key={`scan-line-${lineIndex}`}
                            style={{
                                whiteSpace: 'nowrap',
                                marginBottom: '14px',
                                lineHeight: 1.15
                            }}
                        >
                            {entry.syllables.length === 0 ? (
                                <span>&nbsp;</span>
                            ) : (
                                entry.syllables.map((syllable, syllableIndex) => {
                                    const isPrediction = predictedSet.has(syllableIndex);
                                    const opensPrediction = !hasSegments && isPrediction && !predictedSet.has(syllableIndex - 1);
                                    const closesPrediction = !hasSegments && isPrediction && !predictedSet.has(syllableIndex + 1);
                                    const breakMargin = entry.word_breaks.includes(syllableIndex) ? '0.8em' : '2px';
                                    return (
                                        <React.Fragment key={`syllable-${lineIndex}-${syllableIndex}`}>
                                            {opensPrediction && (
                                                <BracketCell char="[" marginRight="0px" />
                                            )}
                                            <span
                                                style={{
                                                    display: 'inline-block',
                                                    textAlign: 'center',
                                                    marginRight: closesPrediction ? '0px' : breakMargin
                                                }}
                                            >
                                                <span
                                                    style={{
                                                        display: 'block',
                                                        color: MARKER_COLOR,
                                                        fontWeight: 'bold',
                                                        fontSize: '0.75em'
                                                    }}
                                                >
                                                    {entry.markers[syllableIndex]}
                                                </span>
                                                {hasSegments ? (
                                                    <span style={{ display: 'block' }}>
                                                        {entry.segments[syllableIndex].map(([chars, isPredChars], segmentIndex) => {
                                                            const flatIndex = flatOffsets[syllableIndex] + segmentIndex;
                                                            const prevIsPred = flatIndex > 0 && flatSegments[flatIndex - 1][1];
                                                            const nextIsPred = flatIndex < flatSegments.length - 1 && flatSegments[flatIndex + 1][1];
                                                            return (
                                                                <React.Fragment key={`segment-${segmentIndex}`}>
                                                                    {isPredChars && !prevIsPred && (
                                                                        <span style={predictionStyle}>[</span>
                                                                    )}
                                                                    <span style={isPredChars ? predictionStyle : undefined}>
                                                                        {chars}
                                                                    </span>
                                                                    {isPredChars && !nextIsPred && (
                                                                        <span style={predictionStyle}>]</span>
                                                                    )}
                                                                </React.Fragment>
                                                            );
                                                        })}
                                                    </span>
                                                ) : (
                                                    <span
                                                        style={{
                                                            display: 'block',
                                                            color: isPrediction ? MARKER_COLOR : undefined,
                                                            fontWeight: isPrediction ? 'bold' : undefined
                                                        }}
                                                    >
                                                        {syllable}
                                                    </span>
                                                )}
                                            </span>
                                            {closesPrediction && (
                                                <BracketCell char="]" marginRight={breakMargin} />
                                            )}
                                        </React.Fragment>
                                    );
                                })
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export default ScansionDisplay;
