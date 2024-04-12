<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $path = $_POST["path"];

    echo "Path: " . $path . "<br>";
}
?>