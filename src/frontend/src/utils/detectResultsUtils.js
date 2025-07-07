// results color coding logic
export const handleWordColor = (score) => {
    const logScore = Math.log10(score);
    const normalizedScore = Math.max(0, Math.min(1, (logScore + 4) / 4));
    // color-blind-friendly colors
    const colors = [
        '#D55E00', // red
        '#EE7733', // orange
        '#DDAA33', // yellow
        '#228833', // green
    ];
    // map score to color indx
    const colorIndex = Math.floor(normalizedScore * (colors.length - 1));
    return colors[colorIndex];
};