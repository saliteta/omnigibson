document.addEventListener('keydown', function(event) {
    var allowedKeys = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', ' ', 'z', 'q', 'e', 'w', 'a', 's', 'd'];
    var key = event.key;
    if (allowedKeys.includes(key)) {
        fetch('/update_camera', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ key: key })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Reload the images
                var camImage = document.getElementById('camImage');
                camImage.src = '/view_rgb?' + new Date().getTime();

                var depthImage = document.getElementById('depthImage');
                depthImage.src = '/view_depth?' + new Date().getTime();

                var bevImage = document.getElementById('bevImage');
                bevImage.src = '/view_bev?' + new Date().getTime();

                var cameraState = document.getElementById('cameraState');
                cameraState.innerHTML = 'Position: ' + JSON.stringify(data.position) + '<br>' +
                                'Orientation: ' + JSON.stringify(data.orientation);
                // Update camera state text if needed (we'll cover this later)
            }
        })
        .catch(error => console.error('Error:', error));
    }
});
