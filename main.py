import functions_framework


# Global variable, modified within the function by using the global keyword.
count = 0


@functions_framework.http
def discovery(request):
    """
    HTTP Cloud Function that counts how many times it is executed
    within a specific instance.
    Args:
        request (flask.Request): The request object.
        <http://flask.pocoo.org/docs/1.0/api/#flask.Request>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>.
    """
    global count
    count += 1

    # Note: the total function invocation count across
    # all instances may not be equal to this value!
    return f"Instance execution count: {count}"
    